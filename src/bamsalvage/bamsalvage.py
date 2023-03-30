"""
Python script salvages sequences from corrupted BAM files.

"""
import os, sys, re, io
import argparse, struct
import mgzip, gzip
import zlib, logging, json
import numba
import numpy as np
# sys.path.append('/mnt/nas/genomeplatform/scripts/')
# import tkutil

@numba.njit(cache=True)
def convert_bytes_to_seq(buffer:bytes, start:int, l_seq:int, text:bytearray):
    """Numba-acceralated nucleotide converter from 4-bit compressed buffer to a sequence.
    Args:
        buffer (bytes): Unzipped buffer from BGZF block
        start (int): Start position of the sequence
        l_seq (int): Length of the sequence
        text (bytearray): Output buffer in bytes
    """
    bases = [61, 65, 67, 77, 71, 82, 83, 86, 84, 87, 89, 72, 75, 68, 66, 78] # '=ACMGRSVTWYHKDBN'
    j = 0
    end = start + (l_seq + 1) // 2
    for i in range(start, end):
        b = buffer[i]
        text[j] = bases[b >> 4]
        if j + 1 >= l_seq: break
        text[j+1] = bases[b & 15]
        j += 2

@numba.njit(cache=True)
def convert_bytes_to_qual(buffer:bytes, start:int, l_seq:int, text:bytearray):
    """Numba-acceralated QUAL interpreter, simply add 33 to the uchr number.
    Args:
        buffer (bytes): Unzipped buffer from BGZF block
        start (int): Start position of the qual
        l_seq (int): Length of the qual
        text (bytearray): Output buffer in bytes
    """
    j = 0
    end = start + l_seq
    for i in range(start, end):
        b = buffer[i]
        text[i-start] = 33 + b

@numba.njit(cache=True)
def scan_block_header(buffer:bytes, start:numba.int64)->numba.int64:
    """Scanning function of the block header in possible corrupted buffer.

    Args:
        buffer (bytes): BGZF data block possibly lost the entry point.
        start (numba.int64): Initial position of scan.

    Returns:
        numba.int64: Detected start position if possible (-1 will be retruned if no candidates were detected)
    """
    le_I = np.dtype('uint32')#.newbyteorder('<')
    le_i = np.dtype('int32')#.newbyteorder('<')
    for i in range(start, len(buffer)-36):
        block_size = np.frombuffer(buffer[i:i+4], dtype=le_I)[0]
        refid = np.frombuffer(buffer[i+4:i+8], dtype=le_i)[0]
        l_read_name = np.frombuffer(buffer[i+12:i+16], dtype=np.uint8)[0]
        l_seq = np.frombuffer(buffer[i+20:i+24], dtype=le_I)[0]
        if 0 <= l_seq < block_size * 3 // 2 and l_read_name > 1 and -1 <= refid < 200 and block_size + start < len(buffer):
            return i
    return -1

def get_logger(name=None, stdout=True, logfile=None):
    """Logging object  """
    if name is None:
        name = sys._getframe().f_code.co_name
        pass
    
    logger = logging.getLogger(name)
    # set logging
    for h in logger.handlers:
        h.removeHandler(h)
    def _set_log_handler(logger, handler):#, verbose):
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s'))
        logger.addHandler(handler)
        return logger
    if logfile is not None:
        _set_log_handler(logger, logging.FileHandler(logfile))
    else:
        stdout = True
    if stdout:
        _set_log_handler(logger, logging.StreamHandler())
    # logger.setLevel(logging.ERROR)
    logger.propagate = False
    return logger

# @numba.njit('void(u1[:],)')
def read_next_block(handler, **kwargs):
    """Read BGZF block

    Args:
        handler (_io.TextIOWrapper): File handler

    Raises:
        Exception: Gzip header check
        Exception: _description_
        Exception: CRC check
        Exception: Decompressed size check

    Returns:
        _type_: _description_
    """
    logger = kwargs.get('logger', None)
    values = struct.unpack('BBBBIBBH', handler.read(12))

    # GZIP header, ID1=31, ID2=139
    if values[0] != 31 or values[1] != 139:
        if logger:
            logger.warning('GZIP header invalid')
        raise Exception('data block is not gzipped file')
    xlen = values[-1]
    buf = handler.read(xlen)
    si1, si2, slen, bsize = struct.unpack('BBHH', buf[0:6])
    if si1 != 66 or si2 != 67:
        if logger:
            logger.warning('corrupted\n')
        raise Exception('SI1={}, SI2={} should be 66 and 67'.format(si1, si2))
    decobj = zlib.decompressobj(-15) # no header
    compressed = handler.read(bsize - xlen - 19)
    expected_crc = handler.read(4)
    expected_size = struct.unpack('<I', handler.read(4))[0]
    data = decobj.decompress(compressed) + decobj.flush()

    # CRC check
    crc = zlib.crc32(data)
    if crc < 0:
        crc = struct.pack('<i', crc)
    else:
        crc = struct.pack('<I', crc)
    if expected_crc != crc:
        raise Exception('CRC is {:x} , not {:x}'.format(crc, expected_crc))
    if expected_size != len(data):
        sys.stderr.write(f'inconsistent size of decompressed buffer {expected_size} / {len(data)}\n')
    # print('{}:{} ({})=> {} ({})'.format(idx, len(compressed), bsize, expected_size, len(textdata)))
    return data

def retrieve_fastq_from_bam(filename_bam:str, filename_fastq:str, **kwargs)->dict:
    """Retrieving fastq sequences from BAM file

    Args:
        filename_bam (str): BAM filename
        filename_fastq (str): Fastq filename 

    Returns:
        dict: information of results
    """
    logger =kwargs.get('logger', logging.getLogger())
    info = {'input':filename_bam, 'output':filename_fastq}
    force_continuation = kwargs.get('forced', False)
    limit = kwargs.get('limit', 0)
    
    fasta_mode = re.search('\\.m?fa(\\.gz)?$', filename_fastq)
    
    if filename_fastq.endswith('.gz'):                                                                                                                                                   
        n_threads = kwargs.get('threads', 4)
        ostr = io.TextIOWrapper(mgzip.open(filename_fastq, 'wb', thread=n_threads))
    else:
        ostr = open(filename_fastq, 'w')
    
    filesize = os.path.getsize(filename_bam)
    with open(filename_bam, 'rb') as fi:
        idx = 0
        offset = 0
        references = []
        # idx_block = 0
        n_blocks = n_corrupted_blocks = 0
        
        # read header
        try:
            data = read_next_block(fi)
            if data[0:4] != b'BAM\1':
                logger.warning('BAM header lost\n')
                raise Exception('the file is not BAM')
            l_text = struct.unpack('<I', data[4:8])[0]
            while l_text + 12 > len(data):
                sys.stderr.write('reading remant header {}/{}\n'.format(len(data), l_text))
                data += read_next_block(fi)
                n_blocks += 1
            text_ = data[8:8+l_text].decode('latin-1')
            pos = 8 + l_text
            # load references
            n_ref = struct.unpack('<I', data[pos:pos+4])[0]
            logger.info(f'references : {n_ref}')
            pos += 4
            for i in range(n_ref):
                l_name = struct.unpack('<I', data[pos:pos+4])[0]
                pos += 4
                name = data[pos:pos + l_name].decode('latin-1')[:-1]
                pos += l_name
                l_ref = struct.unpack('<I', data[pos:pos+4])[0]
                references.append((name, l_ref))
                # logger.info('@SQ\t{}\tLN:{}'.format(name, l_ref))
                pos += 4
                if pos > len(data): #
                    # logger.info('extend header block {} / {}'.format(pos, len(data)))
                    data += read_next_block(fi)
                    n_blocks += 1
            info['references'] = references
        except Exception as e:
            # failed to decompress given buffer, ignore the block if continuation is set
            if not force_continuation:
                raise
            logger.warning('header was corrupted, skip header blocks')
            n_corrupted_blocks += 1

        # alignment section
        n_seqs = 0
        keep_running = True
        scanning = False
        while keep_running:
            if fi.tell() >= filesize: # check position is in the file size
                break
            try:
                n_blocks += 1
                data = read_next_block(fi)
            except:
                if not force_continuation:
                    raise
                n_corrupted_blocks += 1
                # sys.stderr.write('block {} ({:.1f}% in all) was corrupted.\n'.format(n_blocks, n_corrupted_blocks * 100. / n_blocks))
                scanning = True
                continue # skip the corrupted block
            pos = 0
            if scanning:
                sys.stderr.write('\033[Kscanning {}\r'.format(n_blocks))
                pos_scanned = scan_block_header(data, pos)
                if pos_scanned >= 0:
                    pos = pos_scanned
                    scanning = False
                else: # no candidate position detected, discard current buffer and load next block
                    continue
                    
                # for i in range(len(data) - 36):
                #     block_size, refid, mappos, l_read_name, mapq, bai_bin, n_cigar_op, flag, l_seq, next_refid, next_pos, tlen \
                #         = struct.unpack('<IiiBBHHHIiii', data[pos:pos + 36])
                #     if l_seq < block_size * 3 // 2 and l_read_name > 1 and -1 <= refid < len(references) and block_size < len(data):
                #         pos = i
                #         scanning = False
                #         break
#                if scanning:
#                    continue

            while pos < len(data):
                if pos > 0:
                    data = data[pos:]
                    pos = 0
                block_size, refid, mappos, l_read_name, mapq, bai_bin, n_cigar_op, flag, l_seq, next_refid, next_pos, tlen \
                    = struct.unpack('<IiiBBHHHIiii', data[pos:pos + 36])
                ptr_block_start = pos + 4 # start position of data field
                
                # read data block
                while block_size + ptr_block_start >= len(data):
                    # logger.info('extending alignment block to {} (current {})'.format(block_size, len(data) - pos))
                    try:
                        n_blocks += 1
                        data += read_next_block(fi)
                    except Exception as e:
                        if not force_continuation:
                            raise
                        logger.warning('\033[Kfailed to loading : {}'.format(str(e)))
                        data = []
                        break
#                    print(block_size, len(data))
                if len(data) == 0: # skip blocks
                    n_corrupted_blocks += 1
                    #data = None
                    #pos = 0
                    scanning = True
                    break
                    
                # assert variable range
                if l_seq >= block_size * 3 // 2 or l_read_name == 0 or refid < -2 or refid >= len(references): # invalid block
                    if not force_continuation:
                        raise Exception('invalid field range in {}th seq'.format(n_seqs))
                    sys.stderr.write(f'l_seq={l_seq}, l_read_name={l_read_name}, refid={refid}, block_size={block_size} / {len(data)}\n')
                    #logger.warning('{} th corrupted block in {} : (pos={})'.format(n_corrupted_blocks, n_blocks, pos))
                    n_corrupted_blocks += 1
                    scanning = True
                    data = []
                    pos = 0
                    break
                
                # logger.info(f'pos={pos}/{len(data)}\tname={l_read_name}, l_seq={l_seq}, refid={refid}, pos={mappos}, MAPQ={mapq}, bin={bai_bin}, n_cigar={n_cigar_op}, flag={flag}')
                #bases = '=ACMGRSVTWYHKDBN'
                scanning = False
                pos += 36
                seqid = data[pos:pos + l_read_name].decode('latin-1')[:-1]
                pos += l_read_name
                cigar = data[pos:pos + 4 * n_cigar_op]
                pos += n_cigar_op * 4
                seq = ''

                if fasta_mode: # output only sequence
                    sequence = bytearray(l_seq)
                    convert_bytes_to_seq(data, pos, l_seq, sequence)
                    pos = ptr_block_start + block_size
                    ostr.write('>{}\n{}\n'.format(seqid, sequence.decode('ascii')))
                    n_seqs += 1
                else: # fastq requires sequence and qual
                    sequence = bytearray(l_seq)
                    convert_bytes_to_seq(data, pos, l_seq, sequence)
                    # i_ = 0
                    # for base in data[pos:pos + (l_seq + 1)//2]:
                    #     seq += bases[base >> 4] + bases[base & 15]
                    # if l_seq % 2 == 1: # trim last base for odd length
                    #     seq = seq[:-1]
                        
                    # seq = seq.strip('=')
                    if l_seq > 0:
                        ostr.write('@{}\n{}\n+\n'.format(seqid, sequence.decode('ascii')))
                    pos += (l_seq + 1) // 2
                    if l_seq > 0:
                        convert_bytes_to_qual(data, pos, l_seq, sequence)
                        ostr.write('{}\n'.format(sequence.decode('ascii')))

                    # qual = ''
                    # for v_ in data[pos:pos + l_seq]:
                    #     qual += chr(v_ + 33)
                        
                    # print('{}:{}\t{}\t{}\t{}/{}'.format(n_blocks, pos, seqid, seq[0:20] + '..' + seq[-20:], len(seq), len(qual)))
                    pos += l_seq
                    # ostr.write('@{}\n{}\n+\n{}\n'.format(seqid, seq, qual))
                    n_seqs += 1
                # display current status
                if n_seqs % 1000 == 0:
                    if limit > 0 and n_seqs >= limit:
                        keep_running = False
                        break
                    percentage = fi.tell() / filesize * 100.0
                    sys.stderr.write('\033[K {:.1f}% {}\t{} kreads\t{} blocks ({} corrupted)\r'.format(percentage, seqid[:16], n_seqs // 1000, n_blocks, n_corrupted_blocks))

                block_end = ptr_block_start + block_size

                # auxiliary data 
                if True: 
                    pos = block_end
                    continue
                    
                while pos + 3 < block_end:
                    tag = data[pos:pos + 2].decode('latin-1')
                    val_type = chr(data[pos + 2]) # A:chr, c:i8, C:u8, s:i16, S:U16, i:i32, I:U32, f:float
                    pos += 3
                    if val_type == 'A':
                        pos += 1
                    elif val_type in ('c', 'C'):
                        pos += 1
                    elif val_type in ('s', 'S'):
                        pos += 2
                    elif val_type in ('i', 'I', 'f'):
                        pos += 4
                    elif val_type == 'Z':
                        ptr = start = end = pos
                        while 1:
                            if data[ptr] == 0:
                                end = ptr
                                break
                            ptr += 1
                        # logger.info('{}:{}:{}'.format(tag, val_type, data[start:end]))
                        pos = end + 1
                    elif val_type == 'H':
                        while data[pos] != 0:
                            pos += 2
                    elif val_type == 'B':
                        atype, count_ = struct.unpack('<BI', data[pos:pos+5])
                        pos += 5
                        # print('ARRAY', chr(atype), count_)
                        atype = chr(atype)
                        if atype in ('c', 'C'):
                            pos += count_ 
                        elif atype in ('s', 'S'):
                            pos += count_ * 2
                        elif atype in ('i', 'I', 'f'):
                            pos += count_ * 4
                        while pos > len(data):
                            data += read_next_block(fi)
                            sys.stderr.write('buffer extended to {} / {}\n'.format(pos, len(data)))
                            n_blocks += 1
                    else:
                        if not force_continuation:
                            raise Exception('invalid character {}'.format(int(data[pos+2])))
                        n_corrupted_blocks += 1
                        # raise Exception('invalid value type character {} at {} / {}'.format(ord(val_type), pos, len(data)))
                        break
                    pass
            idx += 1
    ostr.close()
    sys.stderr.write('\033[K\r')
    info['n_seqs'] = n_seqs
    info['n_blocks'] = n_blocks
    info['n_corrupted'] = n_corrupted_blocks 
    return info
    
def main():
    """
    c, b, B, ?, h, H, i, I, l, L, q, Q: 1, 1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 8
    e, f, d : float 2, 4, 8
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+')
    parser.add_argument('-o','--outdir', default='rescued')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--fasta', action='store_true')
    parser.add_argument('--gzip', action='store_true', help='compress output file')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('-p', type=int, default=4, metavar='number', help='Number of threads for gzip compression')
    parser.add_argument('--ignore-corrupted', action='store_true')
    
    args = parser.parse_args()
    outdir = args.outdir
    gzipped = args.gzip
    n_threads = args.p
    os.makedirs(outdir, exist_ok=True)
    limit = args.limit
    filenames = args.input
    forced = args.ignore_corrupted
    fasta = args.fasta

    logger = get_logger(os.path.basename(__file__))
    if args.verbose:
        logger.setLevel(10)
    info = {
        'command':sys.argv,
        'input':filenames,
        'files':[],
    }
    fn_info = os.path.join(outdir, 'run.info')
        
    for filename in filenames:
        if filename.endswith('.bam'):
            title = os.path.basename(filename)[0:-4]
            if fasta:
                filename_out = os.path.join(outdir, title + '.fa')
            else:
                filename_out = os.path.join(outdir, title + '.fastq')
            if gzipped:
                filename_out += '.gz'
            finfo = retrieve_fastq_from_bam(filename, filename_out, logger=logger, limit=limit, forced=forced, threads=n_threads)
            # finfo['filename'] = filename
            info['files'].append(finfo)
            
            with open(fn_info, 'w') as fo:
                json.dump(info, fo, indent=2)
    
if __name__ == '__main__':
    main()
