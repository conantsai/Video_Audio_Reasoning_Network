from PIL import Image
import os

src_folder = "fight_videos/ced/"
tar_folder = "tar"
backup_folder = "backup"


def isCrust(pix):
    return sum(pix) == 0


def hCheck(img, y, step=50):
    count = 0
    width = img.size[0]
    for x in range(0, width, step):
        if isCrust(img.getpixel((x, y))):
            count += 1
        if count > width / step / 2:
            return True
    return False


def vCheck(img, x, step=50):
    count = 0
    height = img.size[1]
    for y in range(0, height, step):
        if isCrust(img.getpixel((x, y))):
            count += 1
        if count > height / step / 2:
            return True
    return False


def boundaryFinder(img, crust_side, core_side, checker):
    if not checker(img, crust_side):
        return crust_side
    if checker(img, core_side):
        return core_side

    mid = (crust_side + core_side) / 2
    while mid != core_side and mid != crust_side:
        if checker(img, mid):
            crust_side = mid
        else:
            core_side = mid
        mid = (crust_side + core_side) / 2
    return core_side


def handleImage(filename):
    try:
        img = Image.open(filename)
    except OSError:
        os.remove(filename)
        return

    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size

    left = boundaryFinder(img, 0, width/2, vCheck)
    right = boundaryFinder(img, width-1, width/2, vCheck)
    top = boundaryFinder(img, 0, height/2, hCheck)
    bottom = boundaryFinder(img, height-1, width/2, hCheck)

    # rect = (left, top, right, bottom)

    if right == width and bottom == height:
        return False
    else:
        return True

    # print(filename, rect)
    # print(width, height)
    # region = img.crop(rect)
    # region.save(filename)


def folderCheck(foldername):
    if foldername:
        if not os.path.exists(foldername):
            os.mkdir(foldername)
            print("Info: Folder \"%s\" created" % foldername)
        elif not os.path.isdir(foldername):
            print("Error: Folder \"%s\" conflict" % foldername)
            return False
    return True


def main():
    # if folderCheck(tar_folder) and folderCheck(src_folder) and folderCheck(backup_folder):
    #     for filename in os.listdir(src_folder):
    #         if filename.split('.')[-1].upper() in ("JPG", "JPEG", "PNG", "BMP", "GIF"):
    #             handleImage(filename, tar_folder)
    #             os.rename(os.path.join(src_folder, filename),
    #                       os.path.join(backup_folder, filename))
    #     pass
    src_list = sorted(os.listdir(src_folder))
    # print(src_list)

    for folder in src_list:
        img_path = os.path.join(src_folder, folder)
        # print(img_path)
        for img in os.listdir(img_path):
            if img.find('jpg') != -1:
                if handleImage(os.path.join(img_path, img)):
                    print(img_path)
                    # print(os.path.join(img_path, img))
                    break


if __name__ == '__main__':
    main()
    # handleImage('Screenshot_2013-10-13-21-55-14.png','')
