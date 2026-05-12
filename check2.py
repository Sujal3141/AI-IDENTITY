import sys

def check_my_stuff():
    print( "python ver ->", sys.version.split(' ')[0] )
    
    # gonna try to grab diffusers real quick
    try:
        import diffusers
        import transformers
        print( "heck yes, diffusers version:", diffusers.__version__ )
        print("transformers looking good too.")
        
    except ImportError as err_thing:
        print("dang it, still broken. error details ->", err_thing)

if __name__ == '__main__':
    check_my_stuff()