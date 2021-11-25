"""

process_file(filename)

  takes templated file .xxx.src and produces .xxx file where .xxx
  is .pyf .f90 or .f using the following template rules:

  '<..>' denotes a template.

  All function and subroutine blocks in a source file with names that
  contain '<..>' will be replicated according to the rules in '<..>'.

  The number of comma-separated words in '<..>' will determine the number of
  replicates.

  '<..>' may have two different forms, named and short. For example,

  named:
   <p=d,s,z,c> where anywhere inside a block '<p>' will be replaced with
   'd', 's', 'z', and 'c' for each replicate of the block.

   <_c>  is already defined: <_c=s,d,c,z>
   <_t>  is already defined: <_t=real,double precision,complex,double complex>

  short:
   <s,d,c,z>, a short form of the named, useful when no <p> appears inside
   a block.

  In general, '<..>' contains a comma separated list of arbitrary
  expressions. If these expression must contain a comma|leftarrow|rightarrow,
  then prepend the comma|leftarrow|rightarrow with a backslash.

  If an expression matches '\\<index>' then it will be replaced
  by <index>-th expression.

  Note that all '<..>' forms in a block must have the same number of
  comma-separated entries.

 Predefined named template rules:
  <prefix=s,d,c,z>
  <ftype=real,double precision,complex,double complex>
  <ftypereal=real,double precision,\\0,\\1>
  <ctype=float,double,complex_float,complex_double>
  <ctypereal=float,double,\\0,\\1>

"""
__all__ = ["process_str", "process_file"]
routine_start_re = ...
routine_end_re = ...
function_start_re = ...

def parse_structure(astr):
    """Return a list of tuples for each function or subroutine each
    tuple is the start and end of a subroutine or function to be
    expanded.
    """
    ...

template_re = ...
named_re = ...
list_re = ...

def find_repl_patterns(astr): ...
def find_and_remove_repl_patterns(astr): ...

item_re = ...

def conv(astr): ...
def unique_key(adict):
    """Obtain a unique key given a dictionary."""
    ...

template_name_re = ...

def expand_sub(substr, names): ...
def process_str(allstr): ...

include_src_re = ...

def resolve_includes(source): ...
def process_file(source): ...

_special_names = ...

def main(): ...

if __name__ == "__main__": ...
