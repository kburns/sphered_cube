"""
Merge distributed analysis sets from a FileHandler.

Usage:
    merge.py <base_path> [--cleanup] [--startlevel=<slev>] [--maxlevel=<mlev>]

Options:
    --cleanup   Delete distributed files after merging
    --startlevel=<mlev>  Start merge level [default: 0]
    --maxlevel=<mlev>  Maximum merge level [default: None]

"""

if __name__ == "__main__":

    from docopt import docopt
    from dedalus.tools import logging
    import post

    args = docopt(__doc__)
    base_path = args['<base_path>']
    cleanup = args['--cleanup']
    startlevel = int(args['--startlevel'])
    maxlevel = args['--maxlevel']
    if maxlevel == 'None':
        maxlevel = None
    else:
        maxlevel = int(maxlevel)

    post.tree_merge_analysis(base_path, cleanup=cleanup, startlevel=startlevel, maxlevel=maxlevel)

