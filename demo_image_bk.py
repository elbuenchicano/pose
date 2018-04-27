import argparse
import cv2
import math
import time
import numpy as np
import util
import json
import os
import sys
import copy
    
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
from utils import u_save2File, u_getPath
from video import video_sequence_by1, video_sequence_byn

keras_weights_file = "model/keras/model.h5"

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


################################################################################
################################################################################
def min_max(a,b):
    if a < b:
        return a, b
    return b, a

################################################################################
################################################################################
def tr_2_0( a ):
    if a < 1:
        return 1
    return a


################################################################################
################################################################################
def process (input_image, params, model_params):

    oriImg = input_image.copy()  # B,G,R order
    #oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    #...........................................................................
    
    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        
        output_blobs = model.predict(input_img)

    

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 0.000000001
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = input_image.copy()  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    stickwidth = 4

    for i in range(17):
    #for i in range(1):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    
    pointsX = []
    pointsY = []

    #print(subset)

    for i in range(1):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            
            pointsX.extend( X )
            pointsY.extend( Y )

    ans = ''
    for i in range(0,len(pointsX),2):
        ans  += str(pointsY[i]) + ' ' + str(pointsX[i]) + ','
        #print (pointsY[i], pointsX[i])
        #cv2.rectangle(canvas, p1, p2, (255,0,0), 2)
        cv2.circle(canvas, (int(pointsY[i]), int(pointsX[i])), radius = 3, color=(255,255,255), thickness=3, lineType=8, shift=0)
        #print(ans)
    ans += '\n'
    #cv2.imshow( 'as', canvas    )
    #cv2.waitKey()        

    return canvas, ans
################################################################################
################################################################################
def detect_image_file(info, params, model_params):
    # generate image with body parts
    file_name   = info['file']
    file_out    = info['out']

    oriImg      = cv2.imread(file_name)  # B,G,R order
    canvas, ans = process(oriImg, params, model_params)

    cv2.imwrite(file_out, canvas)
    u_save2File( os.path.splitext(file_out)[0] + '.txt', ans )

################################################################################
################################################################################
'''
Interface function for video pose detection
'''
def detect_video_file(info, params, model_params):
    file_name   = info['file']
    ini         = info['ini']
    fin         = info['fin']
    out_folder  = info['out_folder']
    visual      = info['visual']
    prop        = info['prop']
    supported   = info['supported']
    step        = info['step']

    
    data        = {}


    out_prop_folder = out_folder + '/props'
    base = os.path.basename(file_name)
    base = os.path.splitext(base)[0] + '.prop'
    name = out_prop_folder + '/' + base

    if os.path.isfile(name):
        print(' File previously processed ')
        return 

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    if step == 1:
        video = video_sequence_by1(file_name, ini, fin)
    else:
        video = video_sequence_byn(file_name, step, ini, fin)

    cap   = cv2.VideoCapture(file_name)

    ret, frame = cap.read()

    print('Reading :' + file_name)
    
    # Iamge video generation ...................................................
    if visual > 0:
        if visual == 1:
            nframe = 0
            while(ret):
                name =  out_folder + '/%05d' % (nframe) + '.png'  
                print('Save: ', name)
                params_c = copy.deepcopy(params)
                model_params_c = copy.deepcopy(model_params)
                canvas, ans = process(frame, params_c, model_params_c)
                cv2.imwrite(name, canvas)
                #ret, frame = video.getCurrent()
                ret, frame = cap.read()
                nframe += 1

        # Video Pose image generation...........................................
        # visual = 2
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            file_n  = os.path.basename(file_name)
            file_n  = os.path.splitext(file_n)[0] + '.avi'
            name_v  =  out_folder + '/' + file_n  

            fps = int(video.cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(name_v,fourcc, fps, (int(video.width), int(video.height)))

            pos = 1
            while(ret):
                print(pos)
                params_c = copy.deepcopy(params)
                model_params_c = copy.deepcopy(model_params)
                canvas, ans = process(frame, params_c, model_params_c)
                out.write(canvas)
                #print('frame')
                #print(video.current-1)
                #ret, frame = video.getCurrent()
                ret, frame = cap.read()
                pos += 1
            
            print('Save video in: ', name_v)
            data['out_video_file'] = name_v
            out.release()

    else:
    # Visual = 0
    # Tracklet file generation.....................................................
        
        print('Tracklet generation only')
        final_ans = ''

        # if you have previous human detection
        if supported:

            

            frame_list     = []
            detection_file  = os.path.splitext( file_name )[0] + '.txt'
            file            = open(detection_file, 'r') 
            
            for line in file: 
                if len(line) < 1:
                    continue
                split_line = line.split(' ')
                frame_list.append(int(split_line[0]))

            frame_list = sorted ( list(set(frame_list)) )
            
            #print(frame_list)
            ret, frame = video.getCurrent()
            
            while ( ret ):
                #print (video.current)
                if (video.current-1) in frame_list:
                    #name =  out_folder + '/%05d' % (video.current - 1) + '.png'  
                    #print('Save: ', name)
        
                    params_c = copy.deepcopy(params)
                    model_params_c = copy.deepcopy(model_params)
                    canvas, ans = process(frame, params_c, model_params_c)
                    
                    if ans is not '\n':
                        final_ans = final_ans + '%05d-' % (video.current-1) + ans

                    #cv2.imwrite(name, canvas)
                    
                    #print(final_ans)

                    #cv2.imshow('frame', canvas)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break
                ret, frame = video.getCurrent()
            
            #print (final_ans)
            base = os.path.basename(file_name)
            base = os.path.splitext(base)[0] + '.txt'
            name =  out_folder + '/' + base
            u_save2File(name, final_ans)
            data['tracklet_file'] = name

        # in complete analisys
        else:
            while(ret):
                
                params_c = copy.deepcopy(params)
                model_params_c = copy.deepcopy(model_params)
                canvas, ans = process(frame, params_c, model_params_c)
                if ans is not '\n':
                    final_ans = final_ans + '%05d-' % (video.current-1) + ans
                ret, frame = video.getCurrent()
                
            
            base = os.path.basename(file_name)
            base = os.path.splitext(base)[0] + '.txt'
            name =  out_folder + '/' + base
            u_save2File(name, final_ans)
            data['tracklet_file'] = name

    #............................................................................
    #proerty flag
    if prop:
        out_prop_folder = out_folder + '/props'
        if not os.path.exists(out_prop_folder):
            os.makedirs(out_prop_folder)

        data_ = {
            "video_file"    : file_name,
            "ini"           : video.pos_ini,
            "fin"           : video.pos_fin,
            "width"         : video.width,
            "height"        : video.height           
            }

        data.update(data_)
        base = os.path.basename(file_name)
        base = os.path.splitext(base)[0] + '.prop'
        name = out_prop_folder + '/' + base

        #saving in file
        print ('Save prop in: ', name)
        with open(name, 'w') as outfile:  
            json.dump(data, outfile)

        
################################################################################
################################################################################
def detect_video_folder(info, params, model_params):
    path        = info['path']
    out_folder  = info['out_folder']
    token       = info['token']
    visual      = info['visual']
    ini         = info['ini']
    fin         = info['fin']
    prop        = info['prop']
    supported   = info['supported']
    
    print('Reading ', path)
    #walking for specific token
    for root, dirs, files in os.walk(path): 
        for file in files:
            if file.endswith(token):
                file_v = os.path.join(root, file)
                name = os.path.splitext(file)[0]
                out_folder_v = out_folder + '/' + name
                
                info_v ={
                    "file"      : file_v,
                    "ini"       : ini,
                    "fin"       : fin,
                    "visual"    : visual,
                    "out_folder": out_folder_v,
                    "supported" : supported,
                    "prop"      : prop
                    }
                detect_video_file(info_v, params, model_params)


################################################################################
################################################################################
if __name__ == '__main__':
    
    funcdict = {'image_file'    : detect_image_file,
                'video_file'    : detect_video_file,
                'video_folder'  : detect_video_folder}
    
    #...........................................................................
    #loading data from configuration file
    confs = json.load(open('pose_configuration.json'))

    tic = time.time()
    print('start processing...')

    # load model
    model = get_testing_model(vgg_norm= False)
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    # calling main function
    funcdict [confs['source_type']](info    = confs[confs['source_type']],
                                   params   = params,
                                   model_params = model_params )
    
    toc = time.time()
    print ('processing time is %.5f' % (toc - tic))


    import argparse
import cv2
import math
import time
import numpy as np
import util
import json
import os
import sys
import copy
    
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
from utils import u_save2File, u_getPath, u_progress, u_init_list_of_objects
from video import video_sequence_by1, video_sequence_byn
from matplotlib import pyplot as plt

keras_weights_file = "model/keras/model.h5"

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


################################################################################
################################################################################
def min_max(a,b):
    if a < b:
        return a, b
    return b, a

################################################################################
################################################################################
def tr_2_0( a ):
    if a < 1:
        return 1
    return a

################################################################################
################################################################################
def predicting_images(images):
    blobs = []
    for i in images:
        data = np.concatenate(i, axis = 0)
        blobs.append( model.predict(data) )
    return blobs

################################################################################
################################################################################
def collecting_images(input_image, params, model_params, buffs, buffs_for):
    oriImg = input_image.copy()  # B,G,R order
    #oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    #...........................................................................
    #speeding up prediction
    
    buffs[0].append(oriImg)
    buffs[1].append(heatmap_avg)
    buffs[2].append(paf_avg)

    for m in range(len(multiplier)):
        scale       = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
        buffs_for[0][m].append(imageToTest)
        buffs_for[1][m].append(imageToTest_padded)
        buffs_for[2][m].append(pad)
        buffs_for[3][m].append(input_img)


    return buffs, buffs_for


################################################################################
################################################################################
def process (buffs, buffs_for, blobs, pos, params, model_params):
    oriImg          = buffs[0][pos]  
    heatmap_avg     = buffs[1][pos]  
    paf_avg         = buffs[2][pos]  
    multiplier  = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    #...........................................................................
    #speeding up prediction
    
    for m in range(len(multiplier)):
        
        imageToTest         = buffs_for[0][m][pos]
        imageToTest_padded  = buffs_for[1][m][pos]
        pad                 = buffs_for[2][m][pos]

        output_blobs        = [ blobs[m][0][pos], blobs[m][1][pos] ] 

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 0.000000001
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = oriImg.copy()  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    stickwidth = 4

    for i in range(17):
    #for i in range(1):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    
    pointsX = []
    pointsY = []

    #print(subset)

    for i in range(1):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            
            pointsX.extend( X )
            pointsY.extend( Y )

    ans = ''
    for i in range(0,len(pointsX),2):
        ans  += str(pointsY[i]) + ' ' + str(pointsX[i]) + ','
        #print (pointsY[i], pointsX[i])
        #cv2.rectangle(canvas, p1, p2, (255,0,0), 2)
        cv2.circle(canvas, (int(pointsY[i]), int(pointsX[i])), radius = 3, color=(255,255,255), thickness=3, lineType=8, shift=0)
        #print(ans)
    ans += '\n'
    #cv2.imshow( 'as', canvas    )
    #cv2.waitKey()        

    return canvas, ans
################################################################################
################################################################################
def detect_image_file(info, params, model_params):
    # generate image with body parts
    file_name   = info['file']
    file_out    = info['out']

    oriImg      = cv2.imread(file_name)  # B,G,R order
    canvas, ans = process(oriImg, params, model_params)

    cv2.imwrite(file_out, canvas)
    u_save2File( os.path.splitext(file_out)[0] + '.txt', ans )


################################################################################
################################################################################
'''
Interface function for video pose detection
'''
def detect_video_file(info, params, model_params):
    file_name   = info['file']
    ini         = info['ini']
    fin         = info['fin']
    out_folder  = info['out_folder']
    visual      = info['visual']
    prop        = info['prop']
    supported   = info['supported']
    step        = info['step']
    buf_size    = info['buf_size']
    
    data        = {}


    out_prop_folder = out_folder + '/props'
    base = os.path.basename(file_name)
    base = os.path.splitext(base)[0] + '.prop'
    name = out_prop_folder + '/' + base

    if os.path.isfile(name):
        print(' File previously processed ')
        return 

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    if step == 1:
        video = video_sequence_by1(file_name, ini, fin)
    else:
        video = video_sequence_byn(file_name, step, ini, fin)

    #cap   = cv2.VideoCapture(file_name)

    ret, frame = video.getCurrent()

    params_c        = copy.deepcopy(params)
    model_params_c  = copy.deepcopy(model_params)
    multiplier = [x * model_params_c['boxsize'] / frame.shape[0] for x in params_c['scale_search']]
    multiplier = len(multiplier)

    print('Reading :' + file_name)
    
    # Iamge video generation ...................................................
    if visual > 0:
        if visual == 1:
            buffs       = [[],[],[]]
            buffs_for   = [ u_init_list_of_objects(multiplier),
                            u_init_list_of_objects(multiplier),
                            u_init_list_of_objects(multiplier),
                            u_init_list_of_objects(multiplier)]
            n_frame = [video.pos_ini]
            
            while(ret):

                u_progress(video.current, video.pos_fin, status= 'Frame: ' + str(video.current-step))
                params_c        = copy.deepcopy(params)
                model_params_c  = copy.deepcopy(model_params)
                buffs, buffs_for= collecting_images(frame, params_c, model_params_c, buffs, buffs_for)
                
                ret, frame      = video.getCurrent()
                n_frame.append(video.current-step)

                if len(buffs[0]) == buf_size:
                    blobs = predicting_images(buffs_for[3])
                    for i in range(buf_size):
                        name =  out_folder + '/%05d' % (n_frame[i]) + '.png'  
                        print('Save: ', name)
                        params_c        = copy.deepcopy(params)
                        model_params_c  = copy.deepcopy(model_params)
                        canvas, ans     = process (buffs, buffs_for, blobs, i, params_c, model_params_c)
                        cv2.imwrite(name, canvas)

                    buffs       = [[],[],[]]
                    buffs_for   = [ u_init_list_of_objects(multiplier),
                                    u_init_list_of_objects(multiplier),
                                    u_init_list_of_objects(multiplier),
                                    u_init_list_of_objects(multiplier)]
                    n_frame = []
            if len(buffs[0]) > 0:
                blobs = predicting_images(buffs_for[3])
                for i in range(len(buffs[0])):
                    name =  out_folder + '/%05d' % (n_frame[i]) + '.png'  
                    print('Save: ', name)
                    params_c        = copy.deepcopy(params)
                    model_params_c  = copy.deepcopy(model_params)
                    canvas, ans     = process (buffs, buffs_for, blobs, i, params_c, model_params_c)
                    cv2.imwrite(name, canvas)

################################################################################
################################################################################
'''
Interface function for video pose detection
'''
def detect_video_file2(info, params, model_params):
    file_name   = info['file']
    ini         = info['ini']
    fin         = info['fin']
    out_folder  = info['out_folder']
    visual      = info['visual']
    prop        = info['prop']
    supported   = info['supported']
    step        = info['step']

    
    data        = {}


    out_prop_folder = out_folder + '/props'
    base = os.path.basename(file_name)
    base = os.path.splitext(base)[0] + '.prop'
    name = out_prop_folder + '/' + base

    if os.path.isfile(name):
        print(' File previously processed ')
        return 

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    if step == 1:
        video = video_sequence_by1(file_name, ini, fin)
    else:
        video = video_sequence_byn(file_name, step, ini, fin)

    cap   = cv2.VideoCapture(file_name)

    ret, frame = cap.read()

    print('Reading :' + file_name)
    
    # Iamge video generation ...................................................
    if visual > 0:
        if visual == 1:
            nframe = 0
            while(ret):
                name =  out_folder + '/%05d' % (nframe) + '.png'  
                print('Save: ', name)
                params_c = copy.deepcopy(params)
                model_params_c = copy.deepcopy(model_params)
                canvas, ans = process(frame, params_c, model_params_c)
                cv2.imwrite(name, canvas)
                #ret, frame = video.getCurrent()
                ret, frame = cap.read()
                nframe += 1

        # Video Pose image generation...........................................
        # visual = 2
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            file_n  = os.path.basename(file_name)
            file_n  = os.path.splitext(file_n)[0] + '.avi'
            name_v  =  out_folder + '/' + file_n  

            fps = int(video.cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(name_v,fourcc, fps, (int(video.width), int(video.height)))

            pos = 1
            while(ret):
                print(pos)
                params_c = copy.deepcopy(params)
                model_params_c = copy.deepcopy(model_params)
                canvas, ans = process(frame, params_c, model_params_c)
                out.write(canvas)
                #print('frame')
                #print(video.current-1)
                #ret, frame = video.getCurrent()
                ret, frame = cap.read()
                pos += 1
            
            print('Save video in: ', name_v)
            data['out_video_file'] = name_v
            out.release()

    else:
    # Visual = 0
    # Tracklet file generation.....................................................
        
        print('Tracklet generation only')
        final_ans = ''

        # if you have previous human detection
        if supported:

            

            frame_list     = []
            detection_file  = os.path.splitext( file_name )[0] + '.txt'
            file            = open(detection_file, 'r') 
            
            for line in file: 
                if len(line) < 1:
                    continue
                split_line = line.split(' ')
                frame_list.append(int(split_line[0]))

            frame_list = sorted ( list(set(frame_list)) )
            
            #print(frame_list)
            ret, frame = video.getCurrent()
            
            while ( ret ):
                #print (video.current)
                if (video.current-1) in frame_list:
                    #name =  out_folder + '/%05d' % (video.current - 1) + '.png'  
                    #print('Save: ', name)
        
                    params_c = copy.deepcopy(params)
                    model_params_c = copy.deepcopy(model_params)
                    canvas, ans = process(frame, params_c, model_params_c)
                    
                    if ans is not '\n':
                        final_ans = final_ans + '%05d-' % (video.current-1) + ans

                    #cv2.imwrite(name, canvas)
                    
                    #print(final_ans)

                    #cv2.imshow('frame', canvas)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break
                ret, frame = video.getCurrent()
            
            #print (final_ans)
            base = os.path.basename(file_name)
            base = os.path.splitext(base)[0] + '.txt'
            name =  out_folder + '/' + base
            u_save2File(name, final_ans)
            data['tracklet_file'] = name

        # in complete analisys
        else:
            while(ret):
                
                params_c = copy.deepcopy(params)
                model_params_c = copy.deepcopy(model_params)
                canvas, ans = process(frame, params_c, model_params_c)
                if ans is not '\n':
                    final_ans = final_ans + '%05d-' % (video.current-1) + ans
                ret, frame = video.getCurrent()
                
            
            base = os.path.basename(file_name)
            base = os.path.splitext(base)[0] + '.txt'
            name =  out_folder + '/' + base
            u_save2File(name, final_ans)
            data['tracklet_file'] = name

    #............................................................................
    #proerty flag
    if prop:
        out_prop_folder = out_folder + '/props'
        if not os.path.exists(out_prop_folder):
            os.makedirs(out_prop_folder)

        data_ = {
            "video_file"    : file_name,
            "ini"           : video.pos_ini,
            "fin"           : video.pos_fin,
            "width"         : video.width,
            "height"        : video.height           
            }

        data.update(data_)
        base = os.path.basename(file_name)
        base = os.path.splitext(base)[0] + '.prop'
        name = out_prop_folder + '/' + base

        #saving in file
        print ('Save prop in: ', name)
        with open(name, 'w') as outfile:  
            json.dump(data, outfile)

        
################################################################################
################################################################################
def detect_video_folder(info, params, model_params):
    path        = info['path']
    out_folder  = info['out_folder']
    token       = info['token']
    visual      = info['visual']
    ini         = info['ini']
    fin         = info['fin']
    prop        = info['prop']
    supported   = info['supported']
    
    print('Reading ', path)
    #walking for specific token
    for root, dirs, files in os.walk(path): 
        for file in files:
            if file.endswith(token):
                file_v = os.path.join(root, file)
                name = os.path.splitext(file)[0]
                out_folder_v = out_folder + '/' + name
                
                info_v ={
                    "file"      : file_v,
                    "ini"       : ini,
                    "fin"       : fin,
                    "visual"    : visual,
                    "out_folder": out_folder_v,
                    "supported" : supported,
                    "prop"      : prop
                    }
                detect_video_file(info_v, params, model_params)


################################################################################
################################################################################
if __name__ == '__main__':
    
    funcdict = {'image_file'    : detect_image_file,
                'video_file'    : detect_video_file,
                'video_folder'  : detect_video_folder}
    
    #...........................................................................
    #loading data from configuration file
    confs = json.load(open('pose_configuration.json'))

    tic = time.time()
    print('start processing...')

    # load model
    model = get_testing_model(vgg_norm= False)
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    # calling main function
    funcdict [confs['source_type']](info    = confs[confs['source_type']],
                                   params   = params,
                                   model_params = model_params )
    
    toc = time.time()
    print ('processing time is %.5f' % (toc - tic))


