import os,sys,json,cv2
from nima.inference.inference_model import InferenceModel
import opt4 as opt
from PIL import Image


# write to result file
def write_json(args,score):
    try:
        outfile =open(args[3],'w')
        #print('saving test json at '+args[3])
    except IndexError:
        print('output_location not found')
        print('saving test json at "./test_data/data.json"')
        outfile = open('./test_data/data.json','w+')
    finally:
        #print(score)
        json.dump(score,outfile)
        outfile.close  


#get score by testing
def test(model,image):
    score = {'results':[]}
    for i in image:
        #'''
        img = cv2.imread(image.get(i))
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        r = model.predict_from_pil_image(img)

        '''
        r = model.predict_from_file(image.get(i))
        '''
        score['results'].append({'name':i, 'score':r['mean_score'], 'b':0, 'c':0})
        #print(r['mean_score'],type(r['mean_score']))
     
    score['results'] = sorted(score['results'], key= lambda x : x['score'], reverse= True)


    return score


#switch mode and execute test
def start(mode,model,image):
    if mode == 'test':
        score = test(model,image)
        '''
        print(str(len(image))+' pictures found!')
        print('=======================')
        print('test_json')
        print(score)
        print('=======================')
        '''
    elif mode == 'adjust':
        #print(image)
        if len(image) > 1:
            print('error adjust more than 1 image')
            sys.exit(0)
        score = test(model,image)
        '''
        print('=======================')
        print('test_json')
        print(score)
        print('=======================')
        '''

        for i in image:
            results = opt.starts(image.get(i))
            #print(i,results)
            filepath, ext = os.path.splitext(image.get(i))
            img_dir = filepath+'_'+ext
            results[2].save(img_dir)
            score['results'].append({'name':i.split('.')[-2]+'_.'+i.split('.')[-1], 'score':results[1], 'b':results[0][0], 'c':results[0][1], 'img_dir':img_dir})
        '''
        print('adjsut_json')
        print(score)
        print('=======================')
        '''
    else:
        print('error select mode')
        sys.exit(0)
    return score


#get image from a folder
def handle_folder(pth):
    image = {}
    for i in os.listdir(pth):
        if i.endswith('.jpeg') or i.endswith('.jpg'):
            image[i] = pth + i
    return image


# detect test or adjust
def detect_mode(args):
    target = args[1]
    if target == 'test':
        return 'test'
    elif target == 'adjust':
        return 'adjust'
    else:
        print('error mode in args[1]')
        sys.exit(0)


# detect folder or a picture
def detect_pth(args):
    target = args[2]
    if target.endswith('.jpeg') or target.endswith('.jpg'):
        try:
            _=open(target)
        except:
            print('error pic')
            sys.exit(0)
        else:
            return 'pic',target
    else:
        try:
            #print(target)
            os.stat(target)
        except:
            print('error folder')
            sys.exit(0)
        else:
            if not target.endswith('/'):
                target += '/'
            try:
                os.stat(target)
            except:
                print('error detecting the path')
                sys.exit(0)
            return 'folder',target


# main
def mynima(args):
    model_pth = './tmp3/emd_loss_epoch_49_train_0.05391903784470127_0.12613263790013726.pth'
    #model_pth = './tmp/emd_loss_epoch_49_train_0.03547421253612805_0.08993149643023331.pth'
    #model_pth = './tmp0710/emd_loss_epoch_49_train_0.06696929275146844_0.1384279955681362.pth'
    model = InferenceModel(path_to_model=model_pth)
    mode = detect_mode(args)
    method,pth = detect_pth(args)
    #print(method)
    if method == 'pic':
        name = os.path.basename(pth)
        #print('name '+name) #get pic name
        image = {name:pth}  #make image dict.
    elif method == 'folder':
        image = handle_folder(pth) #get image dict.
    score = start(mode,model,image)
    write_json(args,score)

    return

#arg[1]: (adjust/test) arg[2]: (folder_path/img_pth) arg[3]: (output_file)
mynima(sys.argv) 