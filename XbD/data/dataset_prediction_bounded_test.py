
"""

Target is in xmin, ymin, xmax, ymax, label
coordinates are in range of [0, 1] normlised height and width

"""

import json, os
import torch
import pdb, time
import torch.utils as tutils
import pickle
import torch.nn.functional as F
import numpy as np
import random as random
from random import shuffle
from PIL import Image

BOUND = 100

def filter_labels(ids, all_labels, used_labels):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label))
    
    return used_ids

def is_part_of_subsets(split_ids, SUBSETS):
    
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    
    return is_it


class VideoDataset(tutils.data.Dataset):
    """
    ROAD Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, train=True, input_type='rgb', transform=None, 
                skip_step=1, full_test=False, explaination=False):

        self.explaination = explaination
        self.ANCHOR_TYPE =  args.ANCHOR_TYPE 
        self.DATASET = args.DATASET
        self.SUBSETS = args.SUBSETS
        self.SEQ_LEN = args.SEQ_LEN
        self.MIN_SEQ_STEP = args.MIN_SEQ_STEP
        self.MAX_SEQ_STEP = args.MAX_SEQ_STEP
        # self.MULIT_SCALE = args.MULIT_SCALE
        self.full_test = full_test
        self.skip_step = skip_step #max(skip_step, self.SEQ_LEN*self.MIN_SEQ_STEP/2)
        self.num_steps = max(1, int(self.MAX_SEQ_STEP - self.MIN_SEQ_STEP + 1 )//2)
        # self.input_type = input_type
        self.input_type = input_type+'-images'
        self.train = train
        self.root = args.DATA_ROOT + args.DATASET + '/'
        self._imgpath = os.path.join(self.root, self.input_type)
        self.prediction_root = args.PREDICTION_ROOT
        self.prediction_dir= os.path.join(self.prediction_root)#change this to our prediction dir
        self.MAX_ANCHOR_BOXES = args.MAX_ANCHOR_BOXES
        self.NUM_CLASSES = args.NUM_CLASSES

        # self.image_sets = image_sets
        self.transform = transform
        self.ids = list()
        if self.DATASET == 'road':
            self._make_lists_road()  
        else:
            raise Exception('Specfiy corect dataset')
        
        self.prediction_db= {}
        for video_name in self.video_list:
            video_id = self.video_list.index(video_name)
            video_path = os.path.join(self.prediction_dir, video_name)
            self.prediction_db[video_id] = {}
            if not os.path.exists(video_path):
                print(f"Prediction directory for video {video_name} does not exist: {video_path}")
                continue
            else:
                self.prediction_db[video_id]["numf"] = BOUND #len(os.listdir(video_path))  # Count the number of frames
                self.prediction_db[video_id]["frames"] = {}
                self.prediction_db[video_id]["boxes"] = {}
                self.prediction_db[video_id]['ego_pred'] = {}
                for i, frame_base_name in enumerate(os.listdir(video_path)):
                    if not frame_base_name.endswith('.pkl'):
                        continue
                    frame_path = os.path.join(video_path, frame_base_name)
                    # Load the predictions from the pickle file
                    with open(frame_path, "rb") as f:
                        orig_preds = pickle.load(f)
                    ego_predictions = orig_preds['ego']
                    agentness = orig_preds['main'][:, 4]
                    boxes = orig_preds['main'][:, :4]  # Bounding boxes
                    preds = orig_preds['main'][:, 5:5+self.NUM_CLASSES] # Remove bounding boxes and agentness

                    # If dim 0 > MAX_ANCHOR_BOXES cut them, else 0 pad them
                    if preds.shape[0] > self.MAX_ANCHOR_BOXES:
                        preds = preds[np.argsort(agentness)][::-1]
                        preds = preds[:self.MAX_ANCHOR_BOXES, :]
                        boxes = boxes[np.argsort(agentness)][::-1]
                        boxes = boxes[:self.MAX_ANCHOR_BOXES, :]
                    elif preds.shape[0] < self.MAX_ANCHOR_BOXES:
                        #   if preds.shape[0] == 0:     shape = (0,200) sono vuoti
                        pad_width = self.MAX_ANCHOR_BOXES - preds.shape[0]
                        preds = np.pad(preds, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
                        boxes = np.pad(boxes, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)

                    #Now prediction_db is a dict with video_name as key
                    # and each video_name has a dict with numf and frames
                    # frames is a dict with frame name as key and predictions as value
                    # frame[:-4] removes the '.pkl' extension
                    self.prediction_db[video_id]["frames"][str(int(frame_base_name[:-4])-1)] = preds
                    self.prediction_db[video_id]["boxes"][str(int(frame_base_name[:-4])-1)] = boxes
                    self.prediction_db[video_id]['ego_pred'][str(int(frame_base_name[:-4])-1)] = ego_predictions
                    if i > BOUND-1:
                        break

        self.num_label_types = len(self.label_types)

    def _make_lists_road(self):

        self.anno_file  = os.path.join(self.root, 'road_trainval_v1.0.json')

        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
        
        database = final_annots['db']
        
        self.label_types =  final_annots['label_types'][:3] #['agent', 'action', 'loc', 'duplex', 'triplet'] #
        
        num_label_type = 3
        self.num_classes = 1 ## one for presence
        self.num_classes_list = [1]
        for name in self.label_types: 
            print('Number of {:s}: all :: {:d} to use: {:d}'.format(name, 
                len(final_annots['all_'+name+'_labels']),len(final_annots[name+'_labels'])))
            numc = len(final_annots[name+'_labels'])
            self.num_classes_list.append(numc)
            self.num_classes += numc
        
        self.ego_classes = final_annots['av_action_labels']
        self.num_ego_classes = len(self.ego_classes)

        self.video_list = []
        self.numf_list = []
        frame_level_list = []

        for videoname in sorted(database.keys()):
            if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS):
                continue
            
            numf = BOUND #database[videoname]['numf']
            self.numf_list.append(numf)
            self.video_list.append(videoname)
            
            frames = database[videoname]['frames']
            frame_level_annos = [ {'labeled':False,'ego_label':-1,'labels':np.asarray([])} for _ in range(numf)]

            frame_nums = [int(f) for f in frames.keys()][:BOUND]
            for frame_num in sorted(frame_nums): #loop from start to last possible frame which can make a legit sequence
                frame_id = str(frame_num)
                if frame_id in frames.keys() and frames[frame_id]['annotated']>0: # == 1
                    
                    frame_index = frame_num-1  
                    frame_level_annos[frame_index]['labeled'] = True 
                    frame_level_annos[frame_index]['ego_label'] = frames[frame_id]['av_action_ids'][0]
                    
                    frame = frames[frame_id]
                    if 'annos' not in frame.keys():
                        frame = {'annos':{}}
                    
                    all_labels = []
                    frame_annos = frame['annos']
                    # Per ogni box in 'annos'
                    for key in frame_annos:
                        anno = frame_annos[key]
                        # anno è una anchor box
                        
                        # Per ogni box crea un vettore di zero con len = self.num_classes (numero totale di classes = 41 + 1)
                        box_labels = np.zeros(self.num_classes)
                        list_box_labels = []
                        cc = 1
                        for idx, name in enumerate(self.label_types):
                            # Quà prendono la annotazione (che può essere action, location, ...)
                            #   e vedono il suo id nelle used_labels e filtrano
                            #   quelle annotazioni che sono in all_labels ma non in used_labels
                            filtered_ids = filter_labels(anno[name+'_ids'], final_annots['all_'+name+'_labels'], final_annots[name+'_labels'])
                            list_box_labels.append(filtered_ids)
                            for fid in filtered_ids:
                                box_labels[fid+cc] = 1
                                box_labels[0] = 1
                            cc += self.num_classes_list[idx+1]

                        # Quà in box_labels avremmo un multi hot encoding per la box
                        # corrente per tutte le possibili classes

                        #list box labels invece ci dà una lista di labels, invece di
                        #   un multi hot encoding

                        all_labels.append(box_labels)
                        # Qui abbiamo aggiunto alla lista all_labels
                        #   che è contestuale al singolo frame
                        #   un vettore di multi hot encoding per la box corrente
                        #   box corrente nominata key

                    all_labels = np.asarray(all_labels, dtype=np.float32)
                    # anche se sono multi hot, mette float (?)
   
                    frame_level_annos[frame_index]['labels'] = all_labels
                    # In frame_level_annos, contestuale al singolo frame,
                    #   mette tutte le annotazioni delle varie bounding
                    #   boxes

            frame_level_list.append(frame_level_annos)  
            # A livello globale mette le frame_level_annos di ogni frame
            #   Poi in un vettore a parte ci sono i video_name
            #   E in un altro vettore il numero di frame per ogni video_name
            #       self.video_list (nomi di video)
            #       self.numf_list  (numero di frame)

            ## make ids
            start_frames = [ f for f in range(numf-self.MIN_SEQ_STEP*self.SEQ_LEN, -1,  -self.skip_step)]
            if 0 not in start_frames:
                start_frames.append(0)
            print('number of start frames: '+ str(len(start_frames)) +  ' video: ' + videoname)
            for frame_num in start_frames:
                # This randomizes frame sampling, from MIN_SEQ_STEP to MAX_SEQ_STEP, 
                #   takes in considerationa step_size only if from the frame_num i am in
                #   can be sampled for SEQ_LEN frames with a skip = step_size without exceeding numf of the video.
                #   
                #   In  our case this does not do anything cause MIN and MAX are both = 1. 
                step_list = [s for s in range(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP+1) if numf-s*self.SEQ_LEN>=frame_num]
                shuffle(step_list)
                # print(len(step_list), self.num_steps)
                for s in range(min(self.num_steps, len(step_list))):
                    video_id = self.video_list.index(videoname)
                    #print(video_id, frame_num, step_list[s])
                    self.ids.append([video_id, frame_num ,step_list[s]])
            # self.ids conterrà per ogni clip:
            #   id del video,   start_frame nel video,
            #       lunghezza della clip

        # pdb.set_trace()
        self.frame_level_list = frame_level_list
        self.all_classes = [['agent_ness']]
        for k, name in enumerate(self.label_types):
            labels = final_annots[name+'_labels']
            self.all_classes.append(labels)
            # self.num_classes_list.append(len(labels))

        self.label_types = ['agent_ness'] + self.label_types
        self.num_videos = len(self.video_list)
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_info = self.ids[index]
        video_id, start_frame, step_size = id_info
        videoname = self.video_list[video_id]
        frame_num = start_frame
        ego_labels = np.zeros(self.SEQ_LEN)-1
        labels = []
        all_boxes = []
        ego_labels = []
        ego_pred = []
        indexs = []
        images = []
        for i in range(self.SEQ_LEN):
            indexs.append(frame_num)
            img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num + 1)
            ego_pred.append(self.prediction_db[video_id]['ego_pred'][str(frame_num)])
            if self.explaination:
                img = Image.open(img_name).convert('RGB')
                images.append(img)
            if self.frame_level_list[video_id][frame_num]['labeled']:
                all_boxes.append(self.prediction_db[video_id]['boxes'][str(frame_num)])
                # Quà dobbiamo prendere le nostre prediction
                labels.append(self.prediction_db[video_id]['frames'][str(frame_num)])
                # Quà prendiamo la gt
                ego_labels.append(self.frame_level_list[video_id][frame_num]['ego_label'])
            else:
                all_boxes.append(self.prediction_db[video_id]['boxes'][str(frame_num)])
                labels.append(self.prediction_db[video_id]['frames'][str(frame_num)])
                
                ego_labels.append(-1)            
            frame_num += step_size
        
        if self.explaination:
            # Convert PIL Images to torch tensors in (C, H, W) format without normalization
            return {
                "images": torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in images]),
                "boxes": torch.tensor(np.array(all_boxes, dtype=np.float32)),
                "labels": torch.tensor(np.array(labels, dtype=np.float32)),
                "ego_labels": torch.tensor(np.array(ego_labels, dtype=np.long))
            }
        else:
            return {
                "labels": torch.tensor(np.array(labels, dtype=np.float32)),
                "ego_labels": torch.tensor(np.array(ego_labels, dtype=np.long)),
                "ego_pred": torch.tensor(np.array(ego_pred, dtype=np.float32))
            }
