import argparse
import glob
import os.path as osp
import xml.etree.ElementTree as ET
from functools import partial

import sys
from collections.abc import Iterable
from multiprocessing import Pool
import numpy as np
from shapely.geometry import Polygon
from shutil import get_terminal_size
import json
import cv2
from time import time


class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(message)


class Timer:
    """A flexible Timer class.
    Examples:
        >>> import time
        >>> import mmcv
        >>> with mmcv.Timer():
        >>>     # simulate a code block that will run for 1s
        >>>     time.sleep(1)
        1.000
        >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
        >>>     # simulate a code block that will run for 1s
        >>>     time.sleep(1)
        it takes 1.0 seconds
        >>> timer = mmcv.Timer()
        >>> time.sleep(0.5)
        >>> print(timer.since_start())
        0.500
        >>> time.sleep(0.5)
        >>> print(timer.since_last_check())
        0.500
        >>> print(timer.since_start())
        1.000
    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    def since_start(self):
        """Total time since the timer is started.
        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.
        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.
        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time() - self._t_last
        self._t_last = time()
        return dur



    
def convert_annotations(image_infos, out_txt_name):
    """Convert the annotation into coco style.

    Args:
        image_infos(list): The list of image information dicts
        out_txt_name(str): The output json filename

    Returns:
        out_json(dict): The coco style dict
    """
    assert isinstance(image_infos, list)
    assert isinstance(out_txt_name, str)
    assert out_txt_name


    with open(out_txt_name, 'w') as out_file:
        for image_info in image_infos:
            img_path = image_info['file_name']
            anno_infos = image_info.pop('anno_info')

            label =[]
            for anno_info in anno_infos:
                points = anno_info['segmentation'][0]
                
                s = []
                for i in range(0, len(points), 2):
                    b = points[i:i + 2]
                    b = [int(t) for t in b]
                    s.append(b)
                result = {"transcription": anno_info['text'], "points": s}
                label.append(result)
            
            out_file.write(img_path + '\t' + json.dumps(
            label, ensure_ascii=False) + '\n')
            


def list_from_file(filename, encoding='utf-8'):
    """Load a text file and parse the content as a list of strings. The
    trailing "\\r" and "\\n" of each line will be removed.

    Note:
        This will be replaced by mmcv's version after it supports encoding.

    Args:
        filename (str): Filename.
        encoding (str): Encoding used to open the file. Default utf-8.

    Returns:
        list[str]: A list of strings.
    """
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for line in f:
            item_list.append(line.rstrip('\n\r'))
    return item_list
class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, '
                            'elapsed: 0s, ETA:')
        else:
            self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
                  f'ETA: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        self.file.flush()
        
def collect_files(img_dir, gt_dir, split):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir(str): The image directory
        gt_dir(str): The groundtruth directory
        split(str): The split of dataset. Namely: training or test

    Returns:
        files(list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    # note that we handle png and jpg only. Pls convert others such as gif to
    # jpg or png offline
    suffixes = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']

    imgs_list = []
    for suffix in suffixes:
        imgs_list.extend(glob.glob(osp.join(img_dir, '*' + suffix)))

    files = []
    if split == 'training':
        for img_file in imgs_list:
            gt_file = gt_dir + '/' + osp.splitext(
                osp.basename(img_file))[0] + '.xml'
            files.append((img_file, gt_file))
        assert len(files), f'No images found in {img_dir}'
        print(f'Loaded {len(files)} images from {img_dir}')
    elif split == 'test':
        for img_file in imgs_list:
            gt_file = gt_dir + '/000' + osp.splitext(
                osp.basename(img_file))[0] + '.txt'
            files.append((img_file, gt_file))
        assert len(files), f'No images found in {img_dir}'
        print(f'Loaded {len(files)} images from {img_dir}')

    return files

def track_progress(func, tasks, bar_width=50, file=sys.stdout, **kwargs):
    """Track the progress of tasks execution with a progress bar.
    Tasks are done with a simple for-loop.
    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        bar_width (int): Width of progress bar.
    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width, file=file)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    prog_bar.file.write('\n')
    return results

def collect_annotations(files, split, nproc=1):
    """Collect the annotation information.

    Args:
        files(list): The list of tuples (image_file, groundtruth_file)
        split(str): The split of dataset. Namely: training or test
        nproc(int): The number of process to collect annotations

    Returns:
        images(list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(split, str)
    assert isinstance(nproc, int)

    load_img_info_with_split = partial(load_img_info, split=split)
    
    images = track_progress(load_img_info_with_split, files)

    return images


def load_txt_info(gt_file, img_info):
    anno_info = []
    for line in list_from_file(gt_file):
        # each line has one ploygen (n vetices), and one text.
        # e.g., 695,885,866,888,867,1146,696,1143,####Latin 9
        line = line.strip()
        strs = line.split(',')
        category_id = 1
        assert strs[28][0] == '#'
        xy = [int(x) for x in strs[0:28]]
        assert len(xy) == 28
        coordinates = np.array(xy).reshape(-1, 2)
        polygon = Polygon(coordinates)
        iscrowd = 0
        area = polygon.area
        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        text = strs[28][4:]

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox,
            area=area,
            text=text,
            segmentation=[xy])
        anno_info.append(anno)
    img_info.update(anno_info=anno_info)
    return img_info


def load_xml_info(gt_file, img_info):

    obj = ET.parse(gt_file)
    anno_info = []
    for image in obj.getroot():  # image
        for box in image:  # image
            h = box.attrib['height']
            w = box.attrib['width']
            x = box.attrib['left']
            y = box.attrib['top']
            text = box[0].text
            segs = box[1].text
            pts = segs.strip().split(',')
            pts = [int(x) for x in pts]
            assert len(pts) == 28
            # pts = []
            # for iter in range(2,len(box)):
            #    pts.extend([int(box[iter].attrib['x']),
            #  int(box[iter].attrib['y'])])
            iscrowd = 0
            category_id = 1
            bbox = [int(x), int(y), int(w), int(h)]

            coordinates = np.array(pts).reshape(-1, 2)
            polygon = Polygon(coordinates)
            area = polygon.area
            anno = dict(
                iscrowd=iscrowd,
                category_id=category_id,
                bbox=bbox,
                area=area,
                text=text,
                segmentation=[pts])
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def load_img_info(files, split):
    """Load the information of one image.

    Args:
        files(tuple): The tuple of (img_file, groundtruth_file)
        split(str): The split of dataset: training or test

    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)
    assert isinstance(split, str)

    img_file, gt_file = files
    img = cv2.imread(img_file)

    
    split_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(split_name, osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        # anno_info=anno_info,
        segm_file=osp.join(split_name, osp.basename(gt_file)))

    if split == 'training':
        img_info = load_xml_info(gt_file, img_info)
    elif split == 'test':
        img_info = load_txt_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ctw1500 annotations to COCO format')
    parser.add_argument('root_path', help='ctw1500 root path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--split-list',
        nargs='+',
        help='a list of splits. e.g., "--split-list training test"')

    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    out_dir = args.out_dir if args.out_dir else root_path

    img_dir = osp.join(root_path, 'imgs')
    gt_dir = osp.join(root_path, 'annotations')

    set_name = {}
    for split in args.split_list:
        set_name.update({split: 'instances_' + split + '.txt'})
        assert osp.exists(osp.join(img_dir, split))

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}') 
        files = collect_files(
            osp.join(img_dir, split), osp.join(gt_dir, split), split)
        image_infos = collect_annotations(files, split, nproc=args.nproc)
        convert_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
