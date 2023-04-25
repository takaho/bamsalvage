import os, sys, re, io
import argparse, struct
import mgzip
import zlib, logging, json
import numba
import numpy as np

# Error handling
class BAMException(Exception):
    """Exception for BAM decoding
    """
    NO_BAM_FILE = 0
    BLOCK_CORRUPTED = 1
    EXCEED_EXPECTED_SIZE = 2
    INCORRECT_MAGIC_NUMBER = 3
    INCORRECT_GZIP_MAGIC_NUMBER = 4
    BUFFER_TERMINATED = 5
    INCONSISTENT_CHECKSUM = 6
    INCONSISTENT_BLOCK_SIZE = 7
    DECOMPRESSION_FAILURE = 8
    OUT_OF_RANGE_VARIABLES = 9
    __KIND = ['This is not BAM file', 
              'Block was corrupted', 'Expected buffer size overflow',
              'Incorrect magic number', 'Incorrect GZIP magic number', 
              'Reached the end of file', 'Inconsistent CRC32 checksum',
              'Decompressed buffer size was inconsisted with expected',
              "Failed to decompress", 'Variables are out of range']
    def __init__(self, kind:int, message:str=''):
        super(BAMException, self).__init__(message)
        if kind < 0 or kind > len(BAMException.__KIND):
            kind = -1
        self.__kind = kind
    def __str__(self):
        return '{} {}'.format(super().__str__(), BAMException.resolve_error_code(self.__kind))
    @classmethod
    def resolve_error_code(cls, kind):
        if 0 <= kind < len(cls.__KIND):
            return cls.__KIND[kind]
        if kind == -1:
            return 'unclassified error in reading a block'
        if kind == -2:
            return 'parameters inconsitency'
        if kind == -3:
            return ''
        return 'Unknown error'
    kind = property(lambda s:s.__kind)

# Optimized functions using Numba
@numba.njit(cache=True)
def convert_bytes_to_seq(buffer:bytes, start:int, l_seq:int, text:bytearray):
    """Convert 4-byte encoded sequence to ASCII
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
    """Convert byte array into QUAL sequence"""
    for i in range(l_seq):
        text[i] = min(33 + buffer[start+i], 127)

@numba.njit(cache=True)
def scan_block_header(buffer:bytes, start:numba.int64)->numba.int64:
    """Scan magic number and signature bytes int data block"""
    # le_I = np.uint32
    # le_i = np.int32
    for i in range(start, len(buffer)-36):
        block_size = np.frombuffer(buffer[i:i+4], dtype=np.uint32)[0]
        refid = np.frombuffer(buffer[i+4:i+8], dtype=np.int32)[0]
        l_read_name = np.frombuffer(buffer[i+12:i+16], dtype=np.uint8)[0]
        l_seq = np.frombuffer(buffer[i+20:i+24], dtype=np.uint32)[0]
        if 0 <= l_seq < block_size * 3 // 2 and l_read_name > 4 and -1 <= refid < 200 and block_size + start < len(buffer):
            return i
    return -1

# @numba.njit('i8(u8[:], u8)', cache=True)
# def _find_byte(bytes, byte):
#     for i in range(bytes.size):
#         if bytes[i] == byte:
#             return i
#     return len(bytes)

def _get_logger(name=None, stdout=True, logfile=None):
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

def scan_next_block(handler, **kwargs):
    """Read BGZF block

     ID1   0-0 u8 = 31 
     ID2   1-1 u8 = 139
     CM    2-2 u8 = 8
     FLG   3-3 u8 = 4
     MTIME 4-7 u32
     XFL   8-8 u8
     OS    9-9 u8
     XLEN  10-11 u16
     SI1    | u8
     SI2    | u8
     SLEN   | u16
     BSIZE  | u16 12-12 + XLEN (min 6)
     CDATA u8[BSIZE-XLEN-19]
     CRC32 u32
     ISIZE u32

    BBBB I BBH BBH H/BII

    Args:
        handler (_io.TextIOWrapper): File handler
        logger : logging.StreamHandler

    Raises:
        BamException: BAM processing error
    Returns:
        _type_: _description_
    """
    # logger = kwargs.get('logger', None)
    # read first 18bytes
    current_pos = handler.tell()
    buf = handler.read(18)
    while buf[0] != 31 or buf[1] != 139 or buf[2] != 8 or buf[3] != 4 or buf[12] != 66 or buf[13] != 67:
        offset = False
        for i in range(1, 18):
            if buf[i] == 31:
                offset = True
                buf = buf[i:] + handler.read(i)
                break
        if not offset:
            buf = handler.read(18)
        if len(buf) < 18: 
            raise BAMException(BAMException.BUFFER_TERMINATED, 'cannot scan header anymore AT {} '.format(handler.tell()))
            
    gzheader = struct.unpack('<BBBBIBBHBBHH', buf)
    xlen = gzheader[7]
    block_size = gzheader[11]
    if xlen >= 6 and block_size > xlen + 19:
        _skip_bytes = xlen - 6
        handler.read(_skip_bytes)
    if len(buf) < xlen:
        raise BAMException(BAMException.BUFFER_TERMINATED, 'cannot load gzipped block AT {} '.format(current_pos))
    compressed_data_size = block_size - xlen - 19
    return decompress_and_validate(handler, compressed_data_size)

    
def read_next_block(handler, **kwargs):
    """Read BGZF block, block corruption is not assumed.

    Args:
        handler (_io.TextIOWrapper): File handler

    Raises:
        BamException: BAM processing error
    Returns:
        _type_: _description_
    """
    # GZIP header, ID1=31, ID2=139
    buf = handler.read(2)
    if len(buf) < 2:
        raise BAMException(BAMException.BUFFER_TERMINATED, 'cannot scan header anymore')
    gzheader = struct.unpack('BB', buf)
    if gzheader[0] != 31 or gzheader[1] != 139:
        raise BAMException(BAMException.INCORRECT_MAGIC_NUMBER)
    buf = handler.read(10)
    values = struct.unpack('<BBIBBH', buf)
    xlen = values[-1]
    buf = handler.read(xlen)
    si1, si2, slen, bsize = struct.unpack('<BBHH', buf[0:6])
    if si1 != 66 or si2 != 67:
        raise BAMException(BAMException.INCORRECT_GZIP_MAGIC_NUMBER, 'SI1={}, SI2={} is different from 66 and 67'.format(si1, si2))
    return decompress_and_validate(handler, bsize - xlen - 19)

def decompress_and_validate(handler, compressed_data_size):
    """Read data block and validate BGZF block. Common routine among read_next_block and scan_next_block functions.

    Args:
        handler (stream): Stream handler
        compressed_data_size (_type_): _description_

    Raises:
        BAMException: Corrupted block detected

    Returns:
        byte array : data block
    """
    current_pos = handler.tell()
    cdata = handler.read(compressed_data_size)

    decobj = zlib.decompressobj(-15) # no header
    try:
        decompressed = decobj.decompress(cdata) + decobj.flush()
    except Exception as e:
        raise BAMException(BAMException.DECOMPRESSION_FAILURE, '{} AT {} '.format(str(e), current_pos))

    buf = handler.read(8)
    if len(buf) < 8:
        raise BAMException(BAMException.BUFFER_TERMINATED, ' AT {} '.format(current_pos))
    crc32, input_size = struct.unpack('<II', buf)
    if input_size == len(decompressed):
        crc32_calc = zlib.crc32(decompressed)
        if crc32_calc != crc32:            
            raise BAMException(BAMException.INCONSISTENT_CHECKSUM, 'CRC is {:x} , not {:x} AT {} '.format(crc32_calc, crc32, current_pos))
        return decompressed
    else:
        # sys.stderr.write(f'inconsistent size of decompressed buffer {expected_size} / {len(data)}\n')
        raise BAMException(BAMException.INCONSISTENT_BLOCK_SIZE, 
                           'inconsistent zlib size expected={}, decompressed={} AT {} '.format(
                                input_size, len(decompressed), current_pos))

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
    # force_continuation = kwargs.get('forced', False)
    limit = kwargs.get('limit', 0)
    seek_pos = kwargs.get('fileseek', 0)

    fasta_mode = filename_fastq is None or re.search('\\.m?fa(\\.gz)?$', filename_fastq)

    if filename_fastq is None:
        # ostr = None
        ostr = sys.stdout
    elif filename_fastq.endswith('.gz'):                                                                                                                                                   
        n_threads = kwargs.get('threads', 4)
        ostr = io.TextIOWrapper(mgzip.open(filename_fastq, 'wb', thread=n_threads))
    else:
        ostr = open(filename_fastq, 'w')
        
    filesize = os.path.getsize(filename_bam)
    tracking_data = []
    # TRACE_ID_READING = 0
    # TRACE_ID_ALIGNMENT = 1
    # tracing_situations = [TRACE_ID_READING, TRACE_ID_ALIGNMENT]
    # tracing_ptr = [-1] * (max(tracing_situations) + 1)

    MAXIMUM_BLOCK_SIZE = 1024 * 1024
    MINIMUM_READ_NAME = 2
    with open(filename_bam, 'rb') as fi:
        if seek_pos > 0:
            logger.info('seek at {}'.format(seek_pos))
            fi.seek(min(seek_pos, filesize))

        references = []
        n_blocks = 0
        pos = 0
        # n_corrupted_blocks = 0
        n_malformed_gzip_blocks = 0
        n_unaligned_blocks = 0
        
        # read header
        if seek_pos == 0: # read header if the process starts from the top
            file_ptr = 0 # pointer of file for backtracking
            try:
                n_blocks += 1
                data = read_next_block(fi)
                if data[0:4] != b'BAM\1':
                    logger.warning('BAM header lost\n')
                    raise Exception('the file is not BAM')
                l_text = struct.unpack('<I', data[4:8])[0]
                while l_text + 12 > len(data):
                    logger.info('reading remant header {}/{}\n'.format(len(data), l_text))
                    n_blocks += 1
                    data += scan_next_block(fi)
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
                        n_blocks += 1
                        data += scan_next_block(fi)
                info['references'] = references
            except BAMException as e:
                if e.kind == BAMException.BUFFER_TERMINATED: # end of file
                    logger.warning('no header and file terminated')
                    return
                else:
                    logger.warning(str(e))
                tracking_data.append([e.kind, file_ptr])
                # logger.warning('header was corrupted, skip header blocks {}'.format(str(e)))
                n_malformed_gzip_blocks += 1
            except Exception as e:
                # if not force_continuation:
                #     raise
                tracking_data.append([-2, file_ptr])
                logger.warning('header was corrupted, skip header blocks')
                n_malformed_gzip_blocks += 1
                raise

        # alignment section
        n_seqs = 0
        total_bases = 0
        keep_running = True
#        scanning = False
        data = []
        # scanning = True
        
        # scanning = True
        while keep_running:
            # first block after buffer flushing
            file_ptr = fi.tell()
            if file_ptr >= filesize: # check position is in the file size
                tracking_data.append([BAMException.BUFFER_TERMINATED, file_ptr]) # end
                break
            try:
                n_blocks += 1
                if pos == 0:
                    data = scan_next_block(fi)
                else:
                    data = read_next_block(fi)
            except BAMException as e:
                logger.warning(str(e))
                n_malformed_gzip_blocks += 1
                tracking_data.append([e.kind, file_ptr])
                # scanning = True
                pos = 0
                continue
            except:
                n_malformed_gzip_blocks += 1
                tracking_data.append([-2, file_ptr])
                # scanning = True
                raise
            pos = 0
            
            # if scanning:
            #     sys.stderr.write('\033[Kscanning {}, {}, {}\r\n'.format(n_blocks, len(data), pos))
            #     pos_scanned = scan_block_header(data, pos)
            #     if pos_scanned >= 0:
            #         pos = pos_scanned
            #         scanning = False
            #     else:
            #         continue
                
            # read data until the end of block
            while pos < len(data):
                file_ptr = fi.tell()
                if file_ptr >= filesize: # check position is in the file size
                    tracking_data.append([BAMException.BUFFER_TERMINATED, file_ptr])
                    break
                if pos > 0:
                    data = data[pos:]
                    pos = 0
                if len(data) < 36: # buffer is empty
                    sys.stderr.write(f'\033[Kbuffer size below 36 AT {file_ptr}\n')
                    data = []
                    pos = 0
                    break
                block_size, refid, mappos, l_read_name, mapq, bai_bin, n_cigar_op, flag, l_seq, next_refid, next_pos, tlen \
                    = struct.unpack('<IiiBBHHHIiii', data[pos:pos + 36])

                # assert variable range
                if l_seq >= block_size * 3 // 2 or l_read_name < 5 or refid < -2 or refid >= len(references): # invalid block
                    n_unaligned_blocks += 1
                    tracking_data.append([BAMException.OUT_OF_RANGE_VARIABLES, file_ptr])
                    data = []
                    pos = 0
                    break

                ptr_block_start = pos + 4 # start position of data field
                
                # print(block_size, l_read_name, l_seq)
                if block_size > MAXIMUM_BLOCK_SIZE or l_read_name < MINIMUM_READ_NAME:
                    block_size = 0
                else:
                    # read data block
                    while block_size + ptr_block_start >= len(data):
                        # print(block_size + ptr_block_start, len(data), fi.tell())
                        # logger.info('extending alignment block to {} (current {})'.format(block_size, len(data) - pos))
                        try:
                            n_blocks += 1
                            data += read_next_block(fi)
                        except BAMException as e:
                            # tracing_ptr[TRACE_ID_READING] = file_ptr
                            tracking_data.append([e.kind, file_ptr])
                            n_malformed_gzip_blocks += 1
                            # sys.stderr.write(str(e) + '\n')
                            logger.warning('\033[Kfailed to loading : {}'.format(str(e)))
                            data = []
                            break
                        except Exception as e:
                            tracking_data.append([-1, file_ptr])
                            raise
                    if len(data) == 0: # skip blocks
                        # scanning = True
                        break
                    
                # logger.info(f'pos={pos}/{len(data)}\tname={l_read_name}, l_seq={l_seq}, refid={refid}, pos={mappos}, MAPQ={mapq}, bin={bai_bin}, n_cigar={n_cigar_op}, flag={flag}')
                #bases = '=ACMGRSVTWYHKDBN'
                # scanning = False
                pos += 36
                seqid = data[pos:pos + l_read_name].decode('latin-1')[:-1]
                pos += l_read_name + n_cigar_op * 4
                n_seqs += 1
                total_bases += l_seq

                if ostr:
                    if fasta_mode:
                        sequence = bytearray(l_seq)
                        convert_bytes_to_seq(data, pos, l_seq, sequence)
                        # seq_ = sequence.decode('ascii')
                        ostr.write('>{}\n{}\n'.format(seqid, sequence.decode('ascii')))
                    else:
                        sequence = bytearray(l_seq)
                        convert_bytes_to_seq(data, pos, l_seq, sequence)
                        if l_seq > 0:
                            ostr.write('@{}\n{}\n+\n'.format(seqid, sequence.decode('ascii')))
                        pos += (l_seq + 1) // 2
                        if l_seq > 0:
                            convert_bytes_to_qual(data, pos, l_seq, sequence)
                            ostr.write('{}\n'.format(sequence.decode('ascii')))
                        # print('{}:{}\t{}\t{}\t{}/{}'.format(n_blocks, pos, seqid, seq[0:20] + '..' + seq[-20:], len(seq), len(qual)))
                        pos += l_seq
                pos = ptr_block_start + block_size
                if n_seqs % 1000 == 0:
                    if limit > 0 and n_seqs >= limit:
                        keep_running = False
                        break
                    percentage = fi.tell() / filesize * 100.0
                    sys.stderr.write('\033[K {:.1f}% {}\t{} kreads\t{} blocks ({} corrupted)\r'.format(
                        percentage, seqid[:16], n_seqs // 1000, n_blocks, len(tracking_data)))

                block_end = ptr_block_start + block_size

                # auxiliary data 
                if True: 
                    pos = block_end
                    continue

                cigar = data[pos:pos + 4 * n_cigar_op]
                pos += n_cigar_op * 4
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
                            try:
                                n_blocks += 1
                                data += read_next_block(fi)
                                sys.stderr.write('buffer extended to {} / {}\n'.format(pos, len(data)))
                            except:
                                tracing_ptr[TRACE_ID_READING] = file_ptr
                                n_malformed_gzip_blocks += 1
                                # if not force_continuation:
                                #     raise
                    else:
                        # if not force_continuation:
                        #     raise Exception('invalid character {}'.format(int(data[pos+2])))
                        n_unaligned_blocks += 1
                        # raise Exception('invalid value type character {} at {} / {}'.format(ord(val_type), pos, len(data)))
                        break
                    pass
    if ostr is not sys.stdout:
        ostr.close()
    sys.stderr.write('\033[K\r')
    info['n_seqs'] = n_seqs
    info['n_blocks'] = n_blocks
    info['n_corrupted'] = n_malformed_gzip_blocks + n_unaligned_blocks
    info['n_malformed_gzip_blocks'] =  n_malformed_gzip_blocks
    info['n_unaligned_blocks'] = n_unaligned_blocks
    info['total_bases'] = total_bases
    info['error.log'] = tracking_data

    # info['traces'] = {'':tracking
    # info['tracing'] = {
    #     'reading':tracing_ptr[TRACE_ID_READING],
    #     'alignment':tracing_ptr[TRACE_ID_ALIGNMENT]}

    return info

# def convert_bam_to_seq(filename_input, outdir, **kwargs):
#     verbose = kwarge.get('verbose', False)
#     mode = kwargs.mode('mode', 'fastq.gz')
#     limit = kwargs.get('limit', 0)
#     n_proc = kwargs.get('n_proc', 4)
#     stop_at_corruption = kwargs.get('stop', False)

def main():
    """Entry point of command "bamsalvage"
    """
    """
    c, b, B, ?, h, H, i, I, l, L, q, Q: 1, 1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 8
    e, f, d : float 2, 4, 8
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+')
    parser.add_argument('-o','--outdir', default=None)
    parser.add_argument('--seek', default=0, type=int, help='seek position, start from given position by percent(0-100, default 0)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mode', choices=['test', 'fasta', 'fasta.gz', 'fastq', 'fastq.gz', 'fa.gz', 'fa', 'fz', 'fq', 'fqz'], default='fastq.gz',
                        help='output mode (test:no output, fq/fastq:fastq fqz/fastq.gz/gzipped fastq, fa/fasta:fasta, fz:gzipped fasta)')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('-p', type=int, default=4, metavar='number', help='Number of threads for gzip compression, this option is ignored if mode is not gzipped output')
    
    args, cmds = parser.parse_known_args()
    if args.input is None:
        bamfiles = []
        for f in cmds:
            if f.endswith('.bam') and os.path.exists(f):
                bamfiles.append(f)
        filenames = bamfiles
    else:
        filenames = args.input
    if len(filenames) == 0:
        parser.print_help(sys.stderr)
        raise Exception('no BAM file specified')
    
    outdir = args.outdir
    n_threads = args.p
    seek_pos = args.seek
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        stdout_mode = False
    else:
        stdout_mode = True
    limit = args.limit
    mode = args.mode
    gzipped = mode in ('fz', 'fastq.gz', 'fqz', 'fasta.gz', 'fa.gz')
    fasta = mode in ('fa', 'fasta', 'fa.gz', 'fz')

    logger = _get_logger(os.path.basename(__file__))
    if args.verbose:
        logger.setLevel(10)
    info = {
        'command':sys.argv,
        'input':filenames,
        'files':[],
    }
    
    if filenames is None:
        raise Exception('no BAM files given')

    for filename in filenames:
        if filename.endswith('.bam'):
            title = os.path.basename(filename)[0:-4]
            if outdir is None or mode == 'test': # no output
                filename_out = None
            elif fasta:
                filename_out = os.path.join(outdir, title + '.fa')
            else:
                filename_out = os.path.join(outdir, title + '.fastq')
            if (not stdout_mode) and gzipped:
                filename_out += '.gz'
            finfo = retrieve_fastq_from_bam(filename, filename_out, logger=logger, limit=limit, threads=n_threads, fileseek=seek_pos)
            tracking_data = finfo.pop('error.log', [])
            filename_log = os.path.join(outdir, '.{}.log'.format(os.path.basename(filename)))
            with open(filename_log, 'w') as fo:
                fo.write('#filename={}\n'.format(filename))
                fo.write('#filesize={}\n'.format(os.path.getsize(filename)))
                fo.write('#code\tfile_pos\tkind\n')
                for kind, pos in tracking_data:
                    fo.write('{}\t{}\t{}\n'.format(kind, pos, BAMException.resolve_error_code(kind)))
            info['files'].append(finfo)
    if outdir is None:
        print(json.dumps(info))
    else:
        fn_info = os.path.join(outdir, 'run.info')
        with open(fn_info, 'w') as fo:
            json.dump(info, fo, indent=2)
    
if __name__ == '__main__':
    main()
