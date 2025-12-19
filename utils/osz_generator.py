import zipfile
import argparse
import os
def create_osz(*paths, output_path=None):
    for path in paths:
        if path.endswith('.osu'):
            osu_path = path
            break
        else:
            osu_path = None
    if output_path is None:
        if osu_path is None:
            raise ValueError("At least one .osu file must be provided if output_path is not specified.")
        output_path = osu_path.rsplit('.', 1)[0] + '.osz'
    else:
        if osu_path is None:
            print("Warning: .osz package created without a .osu file.")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=os.path.basename(path))
    print(f"Created .osz package: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an .osz package from .osu and audio files.")
    file_paths=parser.add_mutually_exclusive_group(required=True)
    file_paths.add_argument('paths', nargs='+',default=None, help="Paths to .osu and audio files to include in the .osz package.")
    file_paths.add_argument('--folder','-f',type=str,default=None,help="If specified, create .osz packages for all files in the given folder.")
    parser.add_argument('--output', '-o', type=str, default="./", help="Output path for the .osz package.")
    args = parser.parse_args()
    if file_paths.folder is not None:
        folder = file_paths.folder
        paths = os.listdir(folder)
        create_osz(*paths, output_path=args.output)
    else:
        create_osz(*args.paths, output_path=args.output)