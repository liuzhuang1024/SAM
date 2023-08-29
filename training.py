import torch
from tqdm import tqdm
from utils import update_lr, Meter, cal_score
from torch import nn
from copy import deepcopy
import pickle
import cv2
import numpy as np


shadow_model = {}

def finetune_part(model: nn.Module, name):
    for k, v in model.named_parameters():
        if name in k:
            v.requires_grad = True
        else:
            v.requires_grad = False
            if k in shadow_model:
                 assert torch.equal(shadow_model[k], v), "find params change!"
            else:
                shadow_model[k] = deepcopy(v)
    return 
    
def train(params, model, optimizer:torch.optim.Optimizer, epoch, train_loader, writer=None):
    model.train()
    device = params['device']
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0

    if params['finetune']:
        finetune_part(model, 'counting')
        
    with tqdm(train_loader, total=len(train_loader)//params['train_parts']) as pbar:
        for batch_idx, (images, image_masks, labels, label_masks, matrix, counting_labels) in enumerate(pbar):
            images, image_masks, labels, label_masks, matrix, counting_labels = \
                images.to(device, non_blocking=True), image_masks.to(device, non_blocking=True), \
                labels.to(device, non_blocking=True), label_masks.to(device, non_blocking=True), \
                matrix.to(device, non_blocking=True), counting_labels.to(device, non_blocking=True)
            batch, time = labels.shape[:2]
            if not 'lr_decay' in params or params['lr_decay'] == 'cosine':
                update_lr(optimizer, epoch, batch_idx, len(train_loader), params['epochs'], params['lr'])
            optimizer.zero_grad()
            probs, loss = model(images, image_masks, labels, label_masks, matrix=matrix, counting_labels=counting_labels)
            word_loss, sim_loss, context_loss, word_state_loss, counting_loss, word_alphas = loss
            loss = word_loss + sim_loss + context_loss + word_state_loss + counting_loss
            loss.backward()

            if params['gradient_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient'])
            optimizer.step()
            loss_meter.add(loss.item())

            wordRate, ExpRate = cal_score(probs, labels, label_masks)
            word_right = word_right + wordRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch
            
            if isinstance(sim_loss, torch.Tensor):
                sim_loss = sim_loss.item()
                
            if writer:
                current_step = epoch * len(train_loader) // params['train_parts'] + batch_idx + 1
                writer.add_scalar('train/loss', loss.item(), current_step)
                writer.add_scalar('train/sim', sim_loss, current_step)
                writer.add_scalar('train/counting', counting_loss, current_step)
                writer.add_scalar('train/context', context_loss, current_step)
                writer.add_scalar('train/word', word_state_loss, current_step)
                writer.add_scalar('train/WordRate', wordRate, current_step)
                writer.add_scalar('train/ExpRate', ExpRate, current_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], current_step)
        
            string = f'{epoch + 1} word_loss:{word_loss.item():.3f}  sim_loss: {sim_loss:.3f} '
            string += f'WRate:{word_right / length:.3f} ERate:{exp_right / cal_num:.3f}'
 
            pbar.set_description(string)
            if batch_idx >= len(train_loader) // params['train_parts']:
                break

        if writer:
            writer.add_scalar('epoch/train_loss', loss_meter.mean, epoch+1)
            writer.add_scalar('epoch/train_WordRate', word_right / length, epoch+1)
            writer.add_scalar('epoch/train_ExpRate', exp_right / cal_num, epoch + 1)
        return loss_meter.mean, word_right / length, exp_right / cal_num


def eval(params, model, epoch, eval_loader, writer=None):
    model.eval()
    device = params['device']
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0
    right_collect = {}
    error_collect = {}
    
    with tqdm(eval_loader, total=len(eval_loader)//params['valid_parts'], disable=True) as pbar, torch.no_grad():
        for batch_idx, (name, images, image_masks, labels, label_masks, matrix, counting_labels) in enumerate(pbar):
            if name[0] not in ['26_em_75']: continue
            # if name[0] not in ['RIT_2014_60']: continue
            
            images, image_masks, labels, label_masks, counting_labels = images.to(device), image_masks.to(
                device), labels.to(device), label_masks.to(device), counting_labels.to(device)
            matrix = matrix.to(device)
            batch, time = labels.shape[:2]
            probs, loss = model(images, image_masks, labels, label_masks, counting_labels=counting_labels, matrix=matrix, is_train=False)
            
            word_loss, sim_loss, _, _, counting_loss, word_alphas = loss
            loss = word_loss + sim_loss
            loss_meter.add(loss.item())
            
            # if torch.equal(probs.argmax(-1), labels):   
            # if 1:   
            #     print(name[0], torch.equal(probs.argmax(-1), labels))
            #     # hot map
            for index, gray_img in enumerate(word_alphas[0].cpu().numpy()):
                gray_img = cv2.resize(gray_img * 255. * 1.5, images.shape[-2:][::-1]).astype(np.uint8)
                heat_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
                # heat_img = cv2.cvtColor(heat_img, cv2.COLOR_RGB2GRAY)
                tmp_image = (255 - images.cpu().numpy()[0, 0]*255).astype(np.uint8)
                tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_GRAY2RGB)
                img_add = cv2.addWeighted(tmp_image, 0.7, heat_img, 0.3, 0)
                cv2.imwrite(f"heatmap/{name[0]}-{index}.jpg", img_add)
            # exit()
            
            # if name[0] in ['RIT_2014_60']:
            #     if torch.equal(probs.argmax(-1), labels):
            #         print(params['val_checkout'], name[0])
            #         exit()
            #     continue
            if torch.equal(probs.argmax(-1), labels):
                print(name[0], eval_loader.dataset.words.decode(probs.argmax(-1)[0, :-1]))
                b = probs.softmax(-1)[0, 5]
                print(torch.topk(b, 5))
            exit()
            if torch.equal(probs.argmax(-1), labels):
                # right_collect[name[0]] = {
                #     "labels": labels[0]
                # }
                right_collect[images.cpu()] = {
                    "labels": labels,
                    "word_out_state_list": model.decoder.word_out_state_list,
                    "word_context_vec_list": model.decoder.word_context_vec_list
                    # "word_out_state_list": model.word_out_state_list,
                    # "word_context_vec_list": model.word_context_vec_list
                }
                
            else:
                error_collect[name[0]] = {
                    # "images": images.cpu(),
                    "labels": labels.cpu(),
                    "prds": probs.argmax(-1).cpu(),
                    # "word_out_state_list": model.decoder.word_out_state_list,
                    # "word_context_vec_list": model.decoder.word_context_vec_list,
                    # "word_out_state_list": model.word_out_state_list,
                    # "word_context_vec_list": model.word_context_vec_list
                }
                
            wordRate, ExpRate = cal_score(probs, labels, label_masks)
            word_right = word_right + wordRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch

            if isinstance(sim_loss, torch.Tensor):
                sim_loss = sim_loss.item()
                
            if writer:
                current_step = epoch * len(eval_loader)//params['valid_parts'] + batch_idx + 1
                writer.add_scalar('eval/word_loss', word_loss.item(), current_step)
                writer.add_scalar('eval/loss', loss.item(), current_step)
                writer.add_scalar('eval/WordRate', wordRate, current_step)
                writer.add_scalar('eval/ExpRate', ExpRate, current_step)

            pbar.set_description(f'{epoch+1} word_loss:{word_loss.item():.4f} sim_loss:{sim_loss:.4f}'
                                 f' WRate:{word_right / length:.4f} ERate:{exp_right / cal_num:.4f}')
            if batch_idx >= len(eval_loader) // params['valid_parts']:
                break
    
        # if writer:
        #     writer.add_scalar('epoch/eval_loss', loss_meter.mean, epoch + 1)
        #     writer.add_scalar('epoch/eval_WordRate', word_right / length, epoch + 1)
        #     writer.add_scalar('epoch/eval_ExpRate', exp_right / len(eval_loader.dataset), epoch + 1)

        # if params['val']:
        #     torch.save(right_collect, "sim_right.pkl")
            # torch.save(error_collect, "can_error.pkl")
        return loss_meter.mean, word_right / length, exp_right / cal_num
        