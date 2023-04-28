# bamsalvage
## INTRODUCTION
bamsalvage is a tools to recover sequence reads as much as possible from possibly corrupted BAM files.
This software share the common purpose with bamrescue by Jérémie Roquet (https://bamrescue.arkanosis.net/). 
bamrescue detects corrupted BGZF block using CRC32 checksums and skip corrupted blocks and the method works well if all blocks begin with new reads.

When we would like to recover long-read sequences, a read can span more than one BGZF blocks since the maximum block size is less than sequencer outputs.

Skipping corrupted blocks does not solve such the troubles and often results in termination of Samtools and failure of sequence recovery.

bamsalvage scans next available start positions when any corrupted blocks are detected.
Since the goal of the software is rescuing sequences, bamsalvage do not recover all information included in BAM file but retrieves reads and qual sequences.

## INSTALL
```
pip install bamsalvage
```

## USAGE
```
bamsalvage -i [FILE] -o {DIRECTORY] --mode [fa,fa.gz,fastq,fastq.gz] [--verbose]
```

```
Options:
  -i, --input <FILE>     Input BAM file
  -o, --output <FILE>    Output filename
  -l, --limit <integer>  Limiting counts [default: 0]
  --split <integer>      Split output file by million reads
  --seek <integer>       Start from given file position
  --not-strict           Skip BGZF block size and CRC32 consistency (not recommended)
  --mode                 Output format (fa:fasta, fa.gz:gzipped fasta, fastq:fastq, fastq.gz(default):gzipped fastq)
  -v, --verbose          verbosity
  -h, --help             Print help
  ```


