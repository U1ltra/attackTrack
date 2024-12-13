import cv2
import glob
import json
from os.path import join, exists
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


def permutations(iterable, max_dist=10):
    # generate indices pairs with distance limitation
    r = 2
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                if abs(indices[1]-indices[0]) > max_dist:
                    break
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

class AttackDataset(Dataset):
    ''' NOTE
    DataLoader cannot batching images with different shape.
    To ensure this ugly code works, keep (batchsize % n_frames)==0 and 
    shuffle=False when setup DataLoader
    ''' 
    def __init__(self, root_dir='data/lasot/person/person-14', frame_sample='random', n_frames=20, test=False):
        # load annotation file
        if 'OTB100' in root_dir: json_fname = 'OTB100.json'
        if 'VOT2019' in root_dir: json_fname = 'VOT2019.json'
        if 'lasot' in root_dir: json_fname = 'anno.json'
        if 'own' in root_dir: json_fname = 'anno.json'
        if 'UAV123' in root_dir: json_fname = 'UAV123.json'
        with open(join(root_dir, json_fname), 'r') as f:
            annos = json.load(f)

        # Only keep person1 data
        if 'person1' in annos:
            annos = {'person1': annos['person1']}  # Keep only person1
        else:
            raise ValueError("person1 not found in the dataset")

        # process video resolutions
        img_shapes = []
        for anno in annos.values():
            img_0 = anno['img_names'][0]
            img_0 = img_0 if 'testing_dataset' in img_0 else join(root_dir, img_0)
            img_shapes.append( cv2.imread(img_0).shape[0:2] )
        unique, counts = np.unique(np.array(img_shapes), return_counts=True, axis=0)
        video_res = unique[counts.argsort()[::-1]]
            
        if test: n_frames = None

        # image name list and gt_bbox list
        self.img_names = list()
        self.bboxs = list()
        video_lens = list()
        for vid_id, anno in annos.items():
            valid_indices = []
            # Get valid frame indices (no NaN in bounding boxes)
            for idx, bbox in enumerate(anno['gt_rect']):
                if not any(np.isnan(x) for x in bbox):
                    valid_indices.append(idx)
                    
            if len(valid_indices) < len(anno['gt_rect']):
                print(f"Warning: Video {vid_id} has {len(anno['gt_rect']) - len(valid_indices)} frames with NaN bboxes")
                
            if not valid_indices:  # Skip video if no valid frames
                continue
                
            if not n_frames:
                # Use all valid frames
                self.img_names.extend([anno['img_names'][i] for i in valid_indices])
                self.bboxs.extend([anno['gt_rect'][i] for i in valid_indices])
                video_lens.append(len(valid_indices))
            else:
                # Randomly sample from valid frames
                sample_size = min(n_frames, len(valid_indices))
                if frame_sample == 'random':
                    chosen_indices = np.random.choice(valid_indices, sample_size, replace=False)
                else:
                    chosen_indices = valid_indices[:sample_size]
                chosen_indices.sort()
                
                self.img_names.extend([anno['img_names'][i] for i in chosen_indices])
                self.bboxs.extend([anno['gt_rect'][i] for i in chosen_indices])
                video_lens.append(len(chosen_indices))

        if not self.img_names:
            raise RuntimeError("No valid frames found in the dataset!")
        
        self.video_seg = np.add.accumulate(video_lens)
        self.video_seg = np.insert(self.video_seg, 0, 0) 
        assert len(self.bboxs) == len(self.img_names)

        # load images to ram if they are not too much
        self.imgs = None
        if len(self.img_names) <= 100:
            if 'data' in self.img_names[0]:
                self.imgs = [np.transpose(cv2.imread(im_name).astype(np.float32), (2, 0, 1)) \
                            for im_name in self.img_names]
            else:
                self.imgs = [np.transpose(cv2.imread(join(root_dir, im_name)).astype(np.float32), (2, 0, 1)) \
                            for im_name in self.img_names]

        self.n_frames = n_frames
        self.test = test
        self.root_dir = root_dir
    
    def gen_ind_combinations(self):
        n_imgs = len(self.img_names)
        list(permutations(n_imgs, 5))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        video_idx = np.searchsorted(self.video_seg, idx, 'right')
        if self.test:
            template_idx = self.video_seg[video_idx-1]
        else:
            # numpy’s RNG is not forkable.
            template_idx = torch.randint(self.video_seg[video_idx-1], self.video_seg[video_idx], (1,)).item()
            
        search_idx = idx
        # print(self.img_names[template_idx], self.img_names[search_idx])
        
        if self.imgs:
            template_img = self.imgs[template_idx]
            search_img = self.imgs[search_idx]
        else:
            template_img_name = self.img_names[template_idx]
            search_img_name = self.img_names[search_idx]
            if 'data' not in template_img_name:
                template_img_name = join(self.root_dir, template_img_name)
            if 'data' not in search_img_name:
                search_img_name = join(self.root_dir, search_img_name)                
            template_img = np.transpose(cv2.imread(template_img_name).astype(np.float32), (2, 0, 1))
            search_img = np.transpose(cv2.imread(search_img_name).astype(np.float32), (2, 0, 1))
        template_bbox = np.array(self.bboxs[template_idx])
        search_bbox = np.array(self.bboxs[search_idx])
       
        # Double check for NaN (shouldn't happen but just in case)
        assert not np.any(np.isnan(template_bbox)), f"NaN found in template_bbox at index {template_idx}"
        assert not np.any(np.isnan(search_bbox)), f"NaN found in search_bbox at index {search_idx}"
   
        return template_img, template_bbox, search_img, search_bbox


if __name__ =='__main__':
    import kornia

    dataset = AttackDataset(n_frames=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=5)

    cv2.namedWindow("template", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("search", cv2.WND_PROP_FULLSCREEN)

    for i in range(20):
        for data in dataloader:
            data = list(map(lambda x: x.split(1), data))
            for template_img, template_bbox, search_img, search_bbox in zip(*data):
                x, y, w, h = template_bbox.squeeze()
                template_img = np.ascontiguousarray(kornia.tensor_to_image(template_img.byte()))
                cv2.rectangle(template_img, (x, y), (x+w, y+h), (0, 0, 255), 4)
                cv2.imshow('template', template_img)
                cv2.waitKey(1)

                x, y, w, h = search_bbox.squeeze()
                search_img = np.ascontiguousarray(kornia.tensor_to_image(search_img.byte()))
                cv2.rectangle(search_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
                cv2.imshow('search', search_img)

                cv2.waitKey(0)