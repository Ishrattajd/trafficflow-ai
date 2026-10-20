import os, shutil, subprocess, sys, urllib.request

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
os.makedirs(RAW, exist_ok=True)
GITHUB = 'https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/'
DRIVE_FOLDER_ID = '10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX'   # DCRNN README Google Drive folder
H5 = os.path.join(RAW, 'metr-la.h5')


def get_github_files():
    for name in ('graph_sensor_ids.txt', 'distances_la_2012.csv', 'graph_sensor_locations.csv'):
        dst = os.path.join(RAW, name)
        if not os.path.exists(dst):
            print('downloading', name)
            urllib.request.urlretrieve(GITHUB + name, dst)


def get_h5():
    if os.path.exists(H5) and os.path.getsize(H5) > 10_000_000:
        print('metr-la.h5 already there')
        return
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
        import gdown
    try:
        files = gdown.download_folder(id=DRIVE_FOLDER_ID, skip_download=True, quiet=True, use_cookies=False)
        target = [f for f in files if os.path.basename(f.path) == 'metr-la.h5'][0]
        print('downloading metr-la.h5 (about 50 MB) ...')
        gdown.download(id=target.id, output=H5, quiet=False)
    except Exception as e:
        print('single-file download failed (', e, '), trying whole folder ...')
        gdown.download_folder(id=DRIVE_FOLDER_ID, output=RAW, quiet=False, use_cookies=False)
        for root, _, names in os.walk(RAW):
            if 'metr-la.h5' in names and os.path.join(root, 'metr-la.h5') != H5:
                shutil.move(os.path.join(root, 'metr-la.h5'), H5)


if __name__ == '__main__':
    get_github_files()
    get_h5()
    ok = os.path.exists(H5) and os.path.getsize(H5) > 10_000_000
    print('metr-la.h5 ready:', ok)
    if not ok:
        print('MANUAL: open https://drive.google.com/open?id=' + DRIVE_FOLDER_ID +
              ' in your browser, download metr-la.h5, put it in data/raw/')