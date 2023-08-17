import time
import torch
import os
import math
import torch.nn as nn
import numpy as np
from tasks.AVE_task import AVE_task
from sklearn.metrics import accuracy_score
from utils.function_tools import save_config,get_device,get_logger,set_seed,weight_init
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



def train(model,train_dataloader,optimizer,scheduler,logger,cfgs,epoch,device,writer,last_epoch_score_a,last_epoch_score_v):
    softmax = nn.Softmax(dim=2)
    model.train()
    model.mode = 'train'
    train_batch_loss = 0.

    if cfgs.modality == "Audio":
        train_score_a = last_epoch_score_a
        accuracy_a = 0.
    elif cfgs.modality == "Visual":
        train_score_v = last_epoch_score_v
        accuracy_v = 0.
    elif cfgs.modality == "Multimodal":
        train_score_a = last_epoch_score_a
        train_score_v = last_epoch_score_v
        accuracy = 0.
        accuracy_a = 0.
        accuracy_v = 0.
    
    nb_batch = train_dataloader.__len__() // cfgs.batch_size
    SHUFFLE_SAMPLES = True
    for step in range(nb_batch):
        audio_inputs, video_inputs, labels, segment_label_batch, segment_avps_gt_batch = train_dataloader.get_batch(step, SHUFFLE_SAMPLES)
        SHUFFLE_SAMPLES = False
        audio_inputs = audio_inputs.to(device)
        video_inputs = video_inputs.to(device)
        labels = labels.long().to(device)
        segment_label_batch = segment_label_batch.to(device)
        segment_avps_gt_batch = segment_avps_gt_batch.to(device)
        logits_labels = labels.argmax(dim=-1)

        iteration = (epoch-1)*nb_batch +step + 1
        if cfgs.modality == "Audio":
            out_a,cross_att = model.net(audio_inputs,video_inputs,cfgs.threshold,pad_audio = False,pad_visual=True)
            loss_cls = nn.CrossEntropyLoss()(out_a.permute(0, 2, 1), segment_label_batch)
            batch_labels = labels.cpu().data.numpy()
            x_a_labels = out_a.cpu().data.numpy()
            batch_accuracy_a = compute_acc(batch_labels,x_a_labels,cfgs.batch_size)
            accuracy_a += batch_accuracy_a / nb_batch
        elif cfgs.modality == "Visual":
            out_v,cross_att = model.net(audio_inputs,video_inputs,cfgs.threshold,pad_audio = True,pad_visual=False)
            loss_cls = nn.CrossEntropyLoss()(out_v.permute(0, 2, 1), segment_label_batch)
            batch_labels = labels.cpu().data.numpy()
            x_v_labels = out_v.cpu().data.numpy()
            batch_accuracy_v = compute_acc(batch_labels,x_v_labels,cfgs.batch_size)
            accuracy_v += batch_accuracy_v / nb_batch
        elif cfgs.modality == "Multimodal":
            out_a,out_v,out,cross_att,cooperation_mean_prob = model(audio_inputs,video_inputs)
            batch_labels = labels.cpu().data.numpy()
            x_labels = out.cpu().data.numpy()
            x_a_labels = out_a.cpu().data.numpy()
            x_v_labels = out_v.cpu().data.numpy()
            batch_accuracy = compute_acc(batch_labels, x_labels, cfgs.batch_size)
            batch_accuracy_a = compute_acc(batch_labels,x_a_labels,cfgs.batch_size)
            batch_accuracy_v = compute_acc(batch_labels,x_v_labels,cfgs.batch_size)
            accuracy += batch_accuracy / nb_batch
            accuracy_a += batch_accuracy_a / nb_batch
            accuracy_v += batch_accuracy_v / nb_batch
            loss_cls = nn.CrossEntropyLoss()(out.permute(0, 2, 1), segment_label_batch) 
            
        loss_avps = AVPSLoss(cross_att, segment_avps_gt_batch)
        loss = loss_cls + cfgs.LAMBDA * loss_avps
        train_batch_loss += loss.item() / nb_batch

        writer.add_scalar('loss/step',loss,iteration - 1)

        optimizer.zero_grad()

        if cfgs.modality == "Audio":
            score_audio = 0.
            num_sample = 0
            for i in range(out_a.size(0)):
                for j in range(out_a.size(1)):
                    num_sample += 1
                    if torch.isinf(torch.log(softmax(out_a)[i][j][logits_labels[i][j]])) or softmax(out_a)[i][j][logits_labels[i][j]] < 1e-8:
                        score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                    else:
                        score_audio += - torch.log(softmax(out_a)[i][j][logits_labels[i][j]])
            score_audio = score_audio / num_sample
            train_score_a = train_score_a * (iteration - 1) / iteration + score_audio.item() / iteration
        elif cfgs.modality == "Visual":
            score_visual = 0.
            num_sample = 0
            for i in range(out_v.size(0)):
                for j in range(out_v.size(1)):
                    num_sample += 1
                    if torch.isinf(torch.log(softmax(out_v)[i][j][logits_labels[i][j]])) or softmax(out_v)[i][j][logits_labels[i][j]] < 1e-8:
                        score_visual += - torch.log(torch.tensor(1e-8,dtype=out_v.dtype,device=out_v.device))
                    else:
                        score_visual += - torch.log(softmax(out_v)[i][j][logits_labels[i][j]])
            score_visual = score_visual / num_sample
            train_score_v = train_score_v * (iteration - 1) / iteration + score_visual.item() / iteration
        elif cfgs.modality == "Multimodal":
            if torch.isnan(out_a).any() or torch.isnan(out_v).any():
                raise ValueError
            
            score_audio = 0.
            score_visual = 0.
            num_sample = 0
            for i in range(out_a.size(0)):
                for j in range(out_a.size(1)):
                    num_sample += 1
                    if torch.isinf(torch.log(softmax(out_a)[i][j][logits_labels[i][j]])) or softmax(out_a)[i][j][logits_labels[i][j]] < 1e-8:
                        score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                    else:
                        score_audio += - torch.log(softmax(out_a)[i][j][logits_labels[i][j]])
                    if torch.isinf(torch.log(softmax(out_v)[i][j][logits_labels[i][j]])) or softmax(out_v)[i][j][logits_labels[i][j]] < 1e-8:
                        score_visual += - torch.log(torch.tensor(1e-8,dtype=out_v.dtype,device=out_v.device))
                    else:
                        score_visual += - torch.log(softmax(out_v)[i][j][logits_labels[i][j]])
            score_audio = score_audio / num_sample
            score_visual = score_visual / num_sample
            ratio_a = math.exp(score_visual.item() - score_audio.item())            
            ratio_v = math.exp(score_audio.item() - score_visual.item())
            
            optimal_ratio_a = math.exp(train_score_v - train_score_a)
            optimal_ratio_v = math.exp(train_score_a - train_score_v)

            coeff_a = math.exp(cfgs.alpha*(optimal_ratio_a - ratio_a))
            coeff_v = math.exp(cfgs.alpha*(optimal_ratio_v - ratio_v))
        
            if cfgs.methods == "AGM" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
                if step ==0:
                    logger.info('Using AGM methods...')
                if cfgs.use_mgpu:
                    model.module.update_scale(coeff_a,coeff_v)
                else:
                    model.update_scale(coeff_a,coeff_v)
            else:
                # normal
                if step ==0:
                    logger.info('Using normal methods...')


            train_score_a = train_score_a * (iteration - 1) / iteration + score_audio.item() / iteration
            train_score_v = train_score_v * (iteration - 1) / iteration + score_visual.item() / iteration
        loss.backward()  

        optimizer.step()
        if step % 100 == 0:
            logger.info('EPOCH:[{:3d}/{:3d}]--STEP:[{:5d}/{:5d}]--{}--Loss:{:.4f}--lr:{}'.format(epoch,cfgs.EPOCHS,step,nb_batch,'Train',loss.item(),[group['lr'] for group in optimizer.param_groups]))
    SHUFFLE_SAMPLES = True

    scheduler.step()
    if cfgs.modality == "Audio":
        writer.add_scalars("Accuracy",{"train_audio_accuracy":accuracy_a},epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]--{}-audio_accuracy:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',accuracy_a))
        return train_score_a,last_epoch_score_v
    elif cfgs.modality == "Visual":
        writer.add_scalars("Accuracy",{"train_visual_accuracy":accuracy_v},epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]--{}-visual_accuracy:{:.4f}'.format(epoch,cfgs.EPOCHS,'Train',accuracy_v))
        return last_epoch_score_a,train_score_v
    else:
        writer.add_scalars('Accuracy(Train)',{'accuracy':accuracy,
                                              'accuracy audio':accuracy_a,
                                              'accuracy visual':accuracy_v},epoch)

        logger.info('EPOCH:[{:3d}/{:3d}]--{}--acc:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}-Alpha:{}'.format(epoch,cfgs.EPOCHS,'Train',accuracy,accuracy_a,accuracy_v,cfgs.alpha))

        return train_score_a,train_score_v
    

def valid(model,valid_dataloader,logger,cfgs,epoch,device,writer):
    softmax = nn.Softmax(dim=2)
    
    with torch.no_grad():
        if cfgs.use_mgpu:
            model.module.eval()
        else:
            model.eval()
        
        if cfgs.modality == "Audio":
            valid_score_a = 0.
            accuracy_a = 0.
        elif cfgs.modality == "Visual":
            valid_score_v = 0.
            accuracy_v = 0.
        else:
            valid_score_a = 0.
            valid_score_v = 0.
            model.mode = "eval"

            accuracy = 0.
            accuracy_a = 0.
            accuracy_v = 0.
        
        nb_batch = valid_dataloader.__len__() // cfgs.batch_size

        SHUFFLE_SAMPLES = False
        for step in range(nb_batch):
            audio_inputs, video_inputs, labels, _,_ = valid_dataloader.get_batch(step, SHUFFLE_SAMPLES)
            audio_inputs = audio_inputs.to(device)
            video_inputs = video_inputs.to(device)
            labels = labels.long().to(device)
            logits_labels = labels.argmax(dim=-1)
            if cfgs.modality == "Audio":
                out_a,cross_att = model.net(audio_inputs,video_inputs,cfgs.threshold,pad_audio = False,pad_visual=True)

                batch_labels = labels.cpu().data.numpy()
                x_a_labels = out_a.cpu().data.numpy()
                batch_accuracy_a = compute_acc(batch_labels,x_a_labels,cfgs.batch_size)
                accuracy_a += batch_accuracy_a / nb_batch

                score_audio = 0.
                num_sample = 0
                for i in range(out_a.size(0)):
                    for j in range(out_a.size(1)):
                        num_sample += 1
                        if torch.isinf(torch.log(softmax(out_a)[i][j][logits_labels[i][j]])) or softmax(out_a)[i][j][logits_labels[i][j]] < 1e-8:
                            score_audio += torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                        else:
                            score_audio += torch.log(softmax(out_a)[i][j][logits_labels[i][j]])
                score_audio = score_audio / num_sample
                valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            elif cfgs.modality == "Visual":
                out_v,cross_att = model.net(audio_inputs,video_inputs,cfgs.threshold,pad_audio = True,pad_visual=False)

                batch_labels = labels.cpu().data.numpy()
                x_v_labels = out_v.cpu().data.numpy()
                batch_accuracy_v = compute_acc(batch_labels,x_v_labels,cfgs.batch_size)
                accuracy_v += batch_accuracy_v / nb_batch
                score_visual = 0.
                num_sample = 0
                for i in range(out_v.size(0)):
                    for j in range(out_v.size(1)):
                        num_sample += 1
                        if torch.isinf(torch.log(softmax(out_v)[i][j][logits_labels[i][j]])) or softmax(out_v)[i][j][logits_labels[i][j]] < 1e-8:
                            score_visual += torch.log(torch.tensor(1e-8,dtype=out_v.dtype,device=out_v.device))
                        else:
                            score_visual += torch.log(softmax(out_v)[i][j][logits_labels[i][j]])
                score_visual = score_visual / num_sample
                valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)
            else:
                out_a,out_v,out,cross_att,cooperation_mean_prob = model(audio_inputs,video_inputs)

                batch_labels = labels.cpu().data.numpy()
                x_labels = out.cpu().data.numpy()
                x_a_labels = out_a.cpu().data.numpy()
                x_v_labels = out_v.cpu().data.numpy()
                batch_accuracy = compute_acc(batch_labels, x_labels, cfgs.batch_size)
                batch_accuracy_a = compute_acc(batch_labels,x_a_labels,cfgs.batch_size)
                batch_accuracy_v = compute_acc(batch_labels,x_v_labels,cfgs.batch_size)
                accuracy += batch_accuracy / nb_batch
                accuracy_a += batch_accuracy_a / nb_batch
                accuracy_v += batch_accuracy_v / nb_batch

                if torch.isnan(out_a).any() or torch.isnan(out_v).any():
                    raise ValueError

                score_audio = 0.
                score_visual = 0.
                num_sample = 0
                for i in range(out_a.size(0)):
                    for j in range(out_a.size(1)):
                        num_sample += 1
                        if torch.isinf(torch.log(softmax(out_a)[i][j][logits_labels[i][j]])) or softmax(out_a)[i][j][logits_labels[i][j]] < 1e-8:
                            score_audio += - torch.log(torch.tensor(1e-8,dtype=out_a.dtype,device=out_a.device))
                        else:
                            score_audio += - torch.log(softmax(out_a)[i][j][logits_labels[i][j]])
                        if torch.isinf(torch.log(softmax(out_v)[i][j][logits_labels[i][j]])) or softmax(out_v)[i][j][logits_labels[i][j]] < 1e-8:
                            score_visual += - torch.log(torch.tensor(1e-8,dtype=out_v.dtype,device=out_v.device))
                        else:
                            score_visual += - torch.log(softmax(out_v)[i][j][logits_labels[i][j]])
                score_audio = score_audio / num_sample
                score_visual = score_visual / num_sample

                valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
                valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)
                ratio_a = math.exp(valid_score_v - valid_score_a)
                ratio_v = math.exp(valid_score_a - valid_score_v)

            if step % 20 == 0:
                logger.info('EPOCH[{:03d}/{:03d}]-STEP:[{:03d}/{:03d}]-{}'.format(epoch,cfgs.EPOCHS,step,nb_batch,'Valid'))
    model.mode = "train"
    
    if cfgs.modality == "Audio":
        writer.add_scalars("Accuracy",{"validate_audio_accuracy":accuracy_a},epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]--{}-audio_accuracy:{:.4f}'.format(epoch,cfgs.EPOCHS,'Validate',accuracy_a))
        return accuracy_a
    elif cfgs.modality == "Visual":
        writer.add_scalars("Accuracy",{"train_visual_accuracy":accuracy_v},epoch)
        logger.info('EPOCH:[{:3d}/{:3d}]--{}-visual_accuracy:{:.4f}'.format(epoch,cfgs.EPOCHS,'Validate',accuracy_v))
        return accuracy_v
    else:
        writer.add_scalars('Accuracy(Valid)',{'accuracy':accuracy,
                                       'accuracy_a':accuracy_a,
                                       'accuracy_v':accuracy_v},epoch)

        logger.info('EPOCH:[{:3d}/{:3d}]--{}--acc:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}'.format(epoch,cfgs.EPOCHS,'Valid',accuracy,accuracy_a,accuracy_v))

        return accuracy

def AVPSLoss(av_simm, soft_label):
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss

def compute_acc(labels, x_labels, nb_batch):
    N = int(nb_batch * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(nb_batch):
        for j in range(x_labels.shape[1]): # x_labels.shape: [bs, 10, 29]
            pre_labels[c] = np.argmax(x_labels[i, j, :]) #
            real_labels[c] = np.argmax(labels[i, j, :])
            c += 1
    target_names = []
    for i in range(29):
        target_names.append("class" + str(i))

    return accuracy_score(real_labels, pre_labels)

def AVE_main(cfgs):
    set_seed(cfgs.random_seed)
    ts = time.strftime('%Y_%m_%d %H:%M:%S',time.localtime())
    save_dir = os.path.join(cfgs.expt_dir,f"{ts}_{cfgs.expt_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_config(cfgs,save_dir)
    logger = get_logger("train_logger",logger_dir=save_dir)
    if cfgs.use_mgpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.gpu_ids
        gpu_ids = list(range(torch.cuda.device_count()))
        device = torch.device('cuda:0')
    else:
        device = get_device(cfgs.device)
    writer = SummaryWriter(os.path.join(save_dir,'tensorboard_out'))
    logger.info(vars(cfgs))
    logger.info(f"Processed ID:{os.getpid()},System Version:{os.uname()}")
    task = AVE_task(cfgs)
    train_dataloader = task.train_dataloader
    valid_dataloader = task.valid_dataloader
    test_dataloader = task.test_dataloader
    
    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler
    model.to(device)

    if cfgs.use_mgpu:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model.cuda()
    best_epoch = {'epoch':0,'acc':0.,'train_loss':0.,'valid_loss':0.}
    start_epoch = 1
    if cfgs.breakpoint_path is not None:
        ckpt = torch.load(cfgs.breakpoint_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_epoch['epoch'] = ckpt['epoch']
        best_epoch['acc'] = ckpt['acc']
        start_epoch = ckpt['epoch'] + 1

    each_epoch_score_a = 0.
    each_epoch_score_v = 0.

    for epoch in range(start_epoch,cfgs.EPOCHS+1):
        logger.info(f'Training for epoch {epoch}...')
        each_epoch_score_a,each_epoch_score_v = train(model,train_dataloader,optimizer,scheduler,logger,cfgs,epoch,device,writer,each_epoch_score_a,each_epoch_score_v)
        logger.info(f'Validating for epoch {epoch}...')
        validate_accuracy = valid(model,valid_dataloader,logger,cfgs,epoch,device,writer)
        if validate_accuracy > best_epoch['acc']:
            best_epoch['acc'] = validate_accuracy
            best_epoch['epoch'] = epoch
            if cfgs.save_checkpoint:
                torch.save({'epoch':epoch,
                            'state_dict':model.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'acc':best_epoch['acc']},os.path.join(save_dir,f'ckpt_full_epoch{epoch}.pth.tar'))
        
        logger.info('Best epoch{},best accuracy{:.4f}'.format(best_epoch["epoch"],best_epoch["acc"]))       