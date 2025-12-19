def create_osz(*paths, output_path=None):
    import zipfile
    import os
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