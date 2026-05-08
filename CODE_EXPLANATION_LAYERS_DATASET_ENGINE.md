"""
================================================================================
DETAILED CODE EXPLANATION - LAYERS, DATASET, ENGINE
================================================================================
"""

# ============================================================================
# 1. MODEL/LAYERS.PY - FPN, DECODER, PROJECTOR
# ============================================================================

class FPN(nn.Module):
    """Feature Pyramid Network - fuses multi-scale vision with text"""
    
    def __init__(self, in_channels=[512, 1024, 1024], out_channels=[256, 512, 1024]):
        """
        in_channels: [C3_in, C4_in, C5_in] = [512, 1024, 1024]
        out_channels: [C3_out, C4_out, C5_out] = [256, 512, 1024]
        """
        super(FPN, self).__init__()
        
        # === TEXT PROJECTION ===
        # Project text feature (1024-dim) to spatial feature
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # (B, 1024) → (B, 1024)
        
        # === FUSION 1: C5 + TEXT ===
        # Fuse highest-level features with text
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        # (B, 1024, 13, 13) → (B, 1024, 13, 13)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        
        # === FUSION 2: C4 + C5_UPSAMPLED ===
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        # (B, 1024, 26, 26) → (B, 512, 26, 26)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                out_channels[1], 1, 0)
        # Cat[(B, 1024, 26, 26), (B, 512, 26, 26)] → (B, 512, 26, 26)
        
        # === FUSION 3: C3 + C4 ===
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        # (B, 512, 52, 52) → (B, 256, 52, 52)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                out_channels[1], 1, 0)
        # Cat[(B, 256, 52, 52), (B, 512, 26, 26)] → (B, 512, 52, 52)
        
        # === AGGREGATION - BACK TO C4 ===
        # Combine all three scales back to 26x26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        # (B, 1024, 13, 13) → (B, 512, 13, 13) → upsample to 26x26
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # Keep (B, 512, 26, 26)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # (B, 512, 52, 52) → downsample info to 26x26
        
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        # Cat[3x(B, 512, 26, 26)] → (B, 512, 26, 26)
        
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))
        # Add coordinate information (pixel positions) to features
    
    def forward(self, imgs, state):
        """
        Args:
            imgs: [(B, 256, 52, 52), (B, 512, 26, 26), (B, 1024, 13, 13)]  # C3, C4, C5
            state: (B, 1024) text global feature
        
        Returns:
            fq: (B, 512, 26, 26) fused multi-modal feature
        """
        v3, v4, v5 = imgs
        
        # === FUSION STEP 1: C5 + TEXT ===
        state_proj = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # (B, 1024, 1, 1)
        f5 = self.f1_v_proj(v5)                                         # (B, 1024, 13, 13)
        f5 = self.norm_layer(f5 * state_proj)                           # Element-wise multiply with text
        # f5: (B, 1024, 13, 13)
        
        # === FUSION STEP 2: C4 + UPSAMPLED C5 ===
        f4 = self.f2_v_proj(v4)                                         # (B, 512, 26, 26)
        f5_up = F.interpolate(f5, scale_factor=2, mode='bilinear')      # (B, 1024, 26, 26)
        f4 = self.f2_cat(torch.cat([f4, f5_up], dim=1))                 # (B, 512, 26, 26)
        
        # === FUSION STEP 3: C3 + C4 ===
        f3 = self.f3_v_proj(v3)                                         # (B, 256, 52, 52)
        f3 = F.avg_pool2d(f3, 2, 2)                                     # (B, 256, 26, 26)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))                    # (B, 512, 26, 26)
        
        # === AGGREGATION: Combine all three scales ===
        fq5 = self.f4_proj5(f5)                                         # (B, 512, 13, 13)
        fq4 = self.f4_proj4(f4)                                         # (B, 512, 26, 26)
        fq3 = self.f4_proj3(f3)                                         # (B, 512, 26, 26)
        
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')       # (B, 512, 26, 26)
        fq = torch.cat([fq3, fq4, fq5], dim=1)                          # (B, 1536, 26, 26)
        fq = self.aggr(fq)                                              # (B, 512, 26, 26)
        fq = self.coordconv(fq)                                         # (B, 512, 26, 26)
        
        return fq


class TransformerDecoder(nn.Module):
    """Cross-attention between vision and text"""
    
    def __init__(self, num_layers, d_model, nhead, dim_ffn, dropout, return_intermediate=False):
        """
        num_layers: 3 decoder layers
        d_model: 512 feature dimension
        nhead: 8 attention heads
        dim_ffn: 2048 feedforward dimension
        """
        super().__init__()
        
        # === DECODER LAYERS ===
        # Stack of 3 transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                   nhead=nhead,
                                   dim_feedforward=dim_ffn,
                                   dropout=dropout) 
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate  # False (return only last)
    
    def forward(self, vis, txt, pad_mask):
        """
        Args:
            vis: (B, 512, 26, 26) visual features from FPN
            txt: (B, 77, 512) text embeddings
            pad_mask: (B, 77) attention mask
        
        Returns:
            output: (B, 512, 676) where 676=26*26
        """
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        
        # === POSITIONAL ENCODING ===
        # Create 2D positional encoding for 26x26 grid
        vis_pos = self.pos2d(C, H, W)                   # (676, 1, 512)
        # Create 1D positional encoding for 77 text tokens
        txt_pos = self.pos1d(D, L)                      # (77, 1, 512)
        
        # === RESHAPE FOR SEQUENCE PROCESSING ===
        # Reshape vision from spatial to sequence format
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)   # (676, B, 512)
        # Reshape text from batch format to sequence format
        txt = txt.permute(1, 0, 2)                      # (77, B, 512)
        
        # === DECODER LAYERS ===
        output = vis                                    # Start with visual features
        for layer in self.layers:
            # Each layer does:
            # 1. Self-attention on visual features
            # 2. Cross-attention: vision attends to text
            # 3. Feed-forward network
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
        
        # === NORMALIZATION ===
        if self.norm is not None:
            # (676, B, 512) → (B, 512, 676)
            output = self.norm(output).permute(1, 2, 0)
            return output
        return output


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention + cross-attention"""
    
    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        """
        Args:
            vis: (HW, B, 512) visual features
            txt: (L, B, 512) text embeddings
            vis_pos: (HW, 1, 512) visual position encoding
            txt_pos: (L, 1, 512) text position encoding
            pad_mask: (B, L) text attention mask
        
        Process:
        1. SELF-ATTENTION: visual features attend to themselves
           - Learn which spatial locations are important
        2. CROSS-ATTENTION: visual features attend to text
           - Learn which text words are relevant to each location
        3. FFN: position-wise feed-forward
           - Non-linear transformation
        """
        
        # === SELF-ATTENTION ===
        # Query = Key = Value = visual features
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)      # Add position info
        vis2 = self.self_attn(q, k, value=vis2)[0]      # Attention
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)                  # Residual connection
        
        # === CROSS-ATTENTION ===
        # Query = visual features, Key = Value = text embeddings
        # Visual pixels learn which text words matter
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(
            query=self.with_pos_embed(vis2, vis_pos),   # Visual query
            key=self.with_pos_embed(txt, txt_pos),      # Text key
            value=txt,                                   # Text value
            key_padding_mask=pad_mask)[0]               # Ignore padding tokens
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)                  # Residual connection
        
        # === FEED-FORWARD NETWORK ===
        # Position-wise MLPs: 512 → 2048 → 512
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)                  # Residual connection
        
        return vis


class Projector(nn.Module):
    """Project to pixel-level predictions using text-guided convolution"""
    
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        """
        word_dim: 1024 (text feature dimension)
        in_dim: 256 (after halving vis_dim=512)
        kernel_size: 3 (3x3 conv kernel)
        """
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        
        # === VISUAL PROCESSING ===
        # Upsample and refine features: 26x26 → 104x104
        self.vis = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),     # 26 → 52
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),     # 52 → 104
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1)
        )
        
        # === TEXT-TO-PIXEL CONVOLUTION ===
        # Text features become convolution parameters
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1  # 256*3*3 + 1 = 2305
        self.txt = nn.Linear(word_dim, out_dim)               # 1024 → 2305
    
    def forward(self, x, word):
        """
        Args:
            x: (B, 512, 26, 26) fused vision features
            word: (B, 1024) text global feature
        
        Returns:
            out: (B, 1, 104, 104) segmentation mask
        
        Process:
        1. Upsample visual features: 26x26 → 104x104
        2. Text → convolution kernel + bias (text-guided conv)
        3. Apply text-guided convolution: text determines pixel predictions
        """
        
        # === PROCESS VISUAL FEATURES ===
        x = self.vis(x)                                  # (B, 256, 104, 104)
        B, C, H, W = x.size()
        # Reshape for grouped convolution: (1, B*256, 104, 104)
        x = x.reshape(1, B * C, H, W)
        
        # === GENERATE TEXT-BASED CONVOLUTION PARAMETERS ===
        # Text → kernel weights + bias
        word_params = self.txt(word)                    # (B, 2305)
        weight, bias = word_params[:, :-1], word_params[:, -1]
        # weight: (B, 2304) → reshape to (B, 256, 3, 3)
        # bias: (B,)
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        
        # === APPLY TEXT-GUIDED CONVOLUTION ===
        # Grouped convolution: each text-guided kernel operates on its corresponding features
        # groups=B means: batch element i gets kernel i
        # (1, B*256, 104, 104) * (B, 256, 3, 3) → (1, B, 104, 104)
        out = F.conv2d(x,
                      weight,
                      padding=self.kernel_size // 2,
                      groups=weight.size(0),
                      bias=bias)
        # Reshape back: (1, B, 104, 104) → (B, 1, 104, 104)
        out = out.transpose(0, 1)
        
        return out


# ============================================================================
# 2. UTILS/DATASET.PY - DATA LOADING
# ============================================================================

class RefDataset(Dataset):
    """Load referring expression segmentation data from LMDB"""
    
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size, word_length):
        """
        lmdb_dir: path to LMDB database with image+text data
        mask_dir: path to segmentation masks
        dataset: 'refcoco', 'refcoco+', etc.
        split: 'train', 'val', 'test'
        mode: 'train', 'val', 'test' (different augmentations)
        input_size: 416 (resize to this size)
        word_length: 77 (pad/truncate text to this length)
        """
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        
        # Normalization statistics (CLIP statistics)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
        
        self.env = None  # LMDB environment (lazily opened)
    
    def _init_db(self):
        """Open LMDB database"""
        self.env = lmdb.open(self.lmdb_dir,
                            subdir=os.path.isdir(self.lmdb_dir),
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))
    
    def __getitem__(self, index):
        """Get single sample from dataset"""
        
        # Lazy initialization of LMDB
        if self.env is None:
            self._init_db()
        
        # Read from LMDB
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        
        # === LOAD AND PROCESS IMAGE ===
        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]  # (H, W)
        
        # === LOAD MASK ===
        seg_id = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        
        # === SELECT RANDOM SENTENCE ===
        idx = np.random.choice(ref['num_sents'])
        sents = ref['sents']
        
        # === GEOMETRIC TRANSFORMATION ===
        # Affine transformation to resize: keep aspect ratio + center crop
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img, mat, self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        
        # === TRAINING MODE ===
        if self.mode == 'train':
            # Load and transform mask
            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8), cv2.IMREAD_GRAYSCALE)
            mask = cv2.warpAffine(mask, mat, self.input_size,
                                 flags=cv2.INTER_LINEAR, borderValue=0.)
            mask = mask / 255.  # Normalize to [0, 1]
            
            # Tokenize sentence to token IDs
            sent = sents[idx]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)  # (word_length,)
            
            # Convert to tensor and normalize
            img, mask = self.convert(img, mask)
            return img, word_vec, mask  # (3,416,416), (77,), (1,416,416)
        
        # === VALIDATION MODE ===
        elif self.mode == 'val':
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            
            params = {
                'mask_dir': mask_dir,
                'inverse': mat_inv,  # For converting predictions back to original size
                'ori_size': np.array(img_size)
            }
            return img, word_vec, params
        
        # === TEST MODE ===
        else:
            img = self.convert(img)[0]
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents  # All sentences for this image
            }
            return img, params
    
    def convert(self, img, mask=None):
        """Convert to tensor and normalize"""
        # RGB → tensor: (H,W,3) → (3,H,W)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        # Convert to float: [0, 255] → [0, 1]
        img = img.float() / 255.0
        # Normalize with CLIP statistics: (x - mean) / std
        img = img.sub_(self.mean).div_(self.std)
        
        if mask is not None:
            mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dim
        
        return (img, mask)


# ============================================================================
# 3. ENGINE/ENGINE.PY - TRAINING AND VALIDATION
# ============================================================================

def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    """Training loop for one epoch"""
    
    # === SETUP METRICS ===
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))
    
    model.train()  # Set to training mode
    time.sleep(2)
    end = time.time()
    
    # === TRAINING LOOP ===
    for i, (image, text, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # === LOAD DATA TO GPU ===
        image = image.cuda(non_blocking=True)  # (B, 3, 416, 416)
        text = text.cuda(non_blocking=True)    # (B, 77)
        target = target.cuda(non_blocking=True).unsqueeze(1)  # (B, 1, 416, 416)
        
        # === FORWARD PASS WITH MIXED PRECISION ===
        with amp.autocast():  # Automatically use lower precision where safe
            pred, target, loss = model(image, text, target)
        
        # === BACKWARD PASS ===
        optimizer.zero_grad()  # Clear previous gradients
        scaler.scale(loss).backward()  # Scaled backprop for mixed precision
        
        # Gradient clipping (prevent exploding gradients)
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        
        scaler.step(optimizer)  # Update weights
        scaler.update()  # Update scale factor
        
        # === COMPUTE METRICS ===
        iou, pr5 = trainMetricGPU(pred, target, 0.35, 0.5)
        
        # === ALL-REDUCE (average across GPUs) ===
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()
        
        # Update meters
        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log to stdout and WandB
        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0]:
                wandb.log({
                    "time/batch": batch_time.val,
                    "time/data": data_time.val,
                    "training/lr": lr.val,
                    "training/loss": loss_meter.val,
                    "training/iou": iou_meter.val,
                    "training/prec@50": pr_meter.val,
                }, step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()  # Don't compute gradients (inference mode)
def validate(val_loader, model, epoch, args):
    """Validation loop"""
    
    iou_list = []
    model.eval()  # Set to evaluation mode
    
    for imgs, texts, param in val_loader:
        # Load data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        
        # === INFERENCE ===
        preds = model(imgs, texts)  # (B, 1, 104, 104)
        preds = torch.sigmoid(preds)  # Convert to probabilities [0, 1]
        
        # === RESIZE TO ORIGINAL SIZE ===
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds,
                                 size=imgs.shape[-2:],
                                 mode='bicubic',
                                 align_corners=True).squeeze(1)
        
        # === COMPUTE IoU FOR EACH SAMPLE ===
        for pred, mask_dir, mat, ori_size in zip(preds, param['mask_dir'],
                                                param['inverse'],
                                                param['ori_size']):
            # Affine transformation back to original image size
            h, w = np.array(ori_size)
            mat = np.array(mat)
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderValue=0.)
            
            # Thresholding: probability → binary mask
            pred = np.array(pred > 0.35)
            
            # Load ground truth mask
            mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.
            
            # === COMPUTE IoU ===
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
    
    # === AGGREGATE METRICS ACROSS GPUS ===
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)  # Gather from all GPUs
    
    # === COMPUTE PRECISION AT DIFFERENT THRESHOLDS ===
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):  # 0.5, 0.6, ..., 0.9
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    
    # === FINAL METRICS ===
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):  # 50%, 60%, ..., 90%
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    
    # Log results
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}'.format(
        epoch, args.epochs, 100. * iou.item())
    logger.info(head + temp)
    
    return iou.item(), prec


# ============================================================================
# KEY METRICS
# ============================================================================

"""
IoU (Intersection over Union):
- Measures overlap between predicted and ground truth masks
- IoU = |pred ∩ gt| / |pred ∪ gt|
- Range: [0, 1], higher is better
- Common threshold: 0.5 (50% overlap is good)

Precision@X:
- Percentage of predictions with IoU > X
- E.g., Pr@50 = percentage of samples with IoU > 0.5
- Example: Pr@50 = 0.85 means 85% of samples have IoU > 0.5

Training Metrics:
- Loss: Binary cross-entropy (lower is better)
- IoU: Intersection over Union
- Prec@50: Accuracy metric (percentage with IoU > 0.5)
"""
