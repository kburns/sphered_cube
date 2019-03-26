"""
Merge distributed analysis sets from a FileHandler.

Usage:
    merge.py <base_path> [--cleanup] [--maxlevel=<lev>]

Options:
    --cleanup   Delete distributed files after merging
    --maxlevel=<lev>  Maximum merge level [default: 0]

"""

if __name__ == "__main__":

    from docopt import docopt
    from dedalus.tools import logging
    import post

    args = docopt(__doc__)
    post.tree_merge_analysis(args['<base_path>'], cleanup=args['--cleanup'], maxlevel=int(args['--maxlevel']))

