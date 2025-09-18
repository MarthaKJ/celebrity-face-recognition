#!/usr/bin/env python3
"""
Helper for evaluation on the Labeled Faces in the Wild (LFW) dataset.

MIT License

Copyright (c) 2016 David Sandberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import datetime
import os
import pickle
import argparse

import mxnet as mx
import numpy as np
import torch
from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import sklearn.preprocessing

from ..backbones.iresnet import iresnet50


class LFold:
    """Simple wrapper around KFold to handle n_splits=1 edge case."""
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            # If n_splits=1, we just return the entire set as train & test
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    """Compute true/false positive rates and accuracy across folds."""
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros(nrof_folds)
    indices = np.arange(nrof_pairs)

    # Pre-compute distance if PCA=0
    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            # PCA if requested
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find best threshold on train_set
        acc_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set]
            )
        best_threshold_index = np.argmax(acc_train)

        # Evaluate on test_set
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set]
            )
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set]
        )

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    """Compute TP/FP/accuracy for given distance threshold."""
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    """Compute validation rate given a FAR target."""
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find threshold that gives FAR = far_target in the training set
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set]
            )
        if np.max(far_train) >= far_target:
            # Remove duplicates from far_train and corresponding thresholds
            unique_far, unique_idx = np.unique(far_train, return_index=True)
            unique_thresholds = thresholds[unique_idx]
            if len(unique_far) < 2:
                threshold = 0.0
            else:
                f_interp = interpolate.interp1d(unique_far, unique_thresholds, kind='slinear',
                                                fill_value="extrapolate")
                threshold = f_interp(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set]
        )

    val_mean = np.mean(val)
    val_std = np.std(val)
    far_mean = np.mean(far)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    """Compute validation rate (VAL) and false accept rate (FAR)."""
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    
    if n_same == 0:
        val = 0.0
    else:
        val = float(true_accept) / float(n_same)
    if n_diff == 0:
        far = 0.0
    else:
        far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    """Top-level evaluation: compute TPR/FPR/ACC + VAL@FAR=1e-3."""
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
        nrof_folds=nrof_folds, pca=pca
    )
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
        1e-3, nrof_folds=nrof_folds
    )
    return tpr, fpr, accuracy, val, val_std, far


@torch.no_grad()
def load_bin(path, image_size):
    """Load LFW (or similar) .bin file and create data tensors for no-flip & flip."""
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # For Python 2
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # For Python 3

    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)

    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))  # HWC -> CHW
        for flip_idx in [0, 1]:
            if flip_idx == 1:
                # Horizontal flip
                flipped = mx.ndarray.flip(data=img, axis=2)
                data_list[flip_idx][idx][:] = torch.from_numpy(flipped.asnumpy())
            else:
                data_list[flip_idx][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)

    print(data_list[0].shape)
    return data_list, issame_list


@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    """Test verification for PyTorch model 'backbone' on given dataset.
    
    Returns:
      - No-flip accuracy (acc1) and std (std1) computed from embeddings_list[0]
      - Flip accuracy (acc2) and std (std2) computed from the sum of embeddings from both flips
      - XNorm value computed across all embeddings
      - embeddings_list: list with embeddings for each flip version
      - A tuple with validation metrics:
          (val_no_flip, val_std_no_flip, far_no_flip, val_flip, val_std_flip, far_flip)
    """
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0

    # For each flip version (0 or 1)
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]  # shape: (batch_size, 3, H, W)
            time0 = datetime.datetime.now()
            img = ((_data / 255.0) - 0.5) / 0.5  # Normalize images to [-1, 1]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            net_out: torch.Tensor = backbone(img.to(device))
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    # Compute XNorm (optional info)
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _norm = np.linalg.norm(embed[i])
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    # Evaluate no-flip accuracy using only embeddings_list[0]
    tpr_nf, fpr_nf, accuracy_nf, val_nf, val_std_nf, far_nf = evaluate(embeddings_list[0], issame_list, nrof_folds=nfolds)
    acc1, std1 = np.mean(accuracy_nf), np.std(accuracy_nf)

    # Evaluate flip accuracy using the sum of embeddings from both flip versions
    embeddings_flip = embeddings_list[0] + embeddings_list[1]
    embeddings_flip = sklearn.preprocessing.normalize(embeddings_flip)
    print(embeddings_flip.shape)
    print('infer time', time_consumed)
    tpr_flip, fpr_flip, accuracy_flip, val_flip, val_std_flip, far_flip = evaluate(embeddings_flip, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy_flip), np.std(accuracy_flip)

    return acc1, std1, acc2, std2, _xnorm, embeddings_list, (val_nf, val_std_nf, far_nf, val_flip, val_std_flip, far_flip)


def dumpR(data_set, mxnet_model, batch_size, name=''):
    """(Optional) Dump verification embeddings using MXNet model (not PyTorch)."""
    print('dump verification embedding..')
    data_list, issame_list = data_set
    embeddings_list = []
    time_consumed = 0.0

    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = mx.nd.array(data[bb - batch_size: bb].numpy(), mxnet_model._context)
            db = mx.io.DataBatch(data=[_data], label=None)
            time0 = datetime.datetime.now()
            mxnet_model.forward(db, is_train=False)
            net_out = mxnet_model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    outname = os.path.join('temp.bin')
    with open(outname, 'wb') as f:
        pickle.dump((embeddings, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Embeddings dumped to {outname}.")


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='do verification')
        parser.add_argument('--data-dir',
                            default='/workspace/datasets/manually-annotated/data',
                            help='Path to your dataset folder (containing .bin files)')
        parser.add_argument('--model',
                            default='/workspace/src/insightface/recognition/arcface_torch/work_dirs/new/model.pt',
                            help='Path to load model (either .pt for PyTorch or prefix for MXNet).')
        parser.add_argument('--target',
                            default='lfw,cfp_ff,cfp_fp,agedb_30,val',
                            help='Comma-separated list of test targets, e.g. lfw, cfp_ff, cfp_fp, agedb_30.')
        parser.add_argument('--gpu', default=0, type=int, help='GPU id for MXNet context (if using MXNet).')
        parser.add_argument('--batch-size', default=32, type=int, help='Batch size for forward passes.')
        parser.add_argument('--max', default='', type=str, help='(Unused optional parameter).')
        parser.add_argument('--mode', default=0, type=int,
                            help='0: verification; else for embedding dump etc.')
        parser.add_argument('--nfolds', default=10, type=int,
                            help='Number of cross-validation folds (usually 10).')
        args = parser.parse_args()

        image_size = [112, 112]
        print('image_size', image_size)

        ctx = mx.gpu(args.gpu)  # Used by MXNet if needed
        nets = []
        prefix = args.model.split(',')[0]

        if prefix.endswith('.pt'):
            print(f"Detected PyTorch model (.pt). Loading state_dict...")
            model = iresnet50(num_features=512, dropout=0.4)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(prefix, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            nets.append(model)
        else:
            pdir = os.path.dirname(prefix)
            print(f"Searching for MXNet .params in: {pdir}")
            epochs = []
            for fname in os.listdir(pdir):
                if not fname.endswith('.params'):
                    continue
                _file = os.path.join(pdir, fname)
                if _file.startswith(prefix):
                    try:
                        epoch = int(fname.split('.')[0].split('-')[1])
                        epochs.append(epoch)
                    except Exception as e:
                        print(f"Failed to extract epoch from {fname}: {e}")
            epochs = sorted(epochs, reverse=True)
            print('model number', len(epochs))
            if len(epochs) == 0:
                raise ValueError("No MXNet model found. Are you sure it's a PyTorch .pt or correct MXNet prefix?")
            time0 = datetime.datetime.now()
            for epoch in epochs:
                print('loading', prefix, epoch)
                sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
                all_layers = sym.get_internals()
                sym = all_layers['fc1_output']
                model_mx = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
                model_mx.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
                model_mx.set_params(arg_params, aux_params)
                nets.append(model_mx)
            time_now = datetime.datetime.now()
            diff = time_now - time0
            print('model loading time', diff.total_seconds())

        ver_list = []
        ver_name_list = []
        print(f"Targets: {args.target}")
        for name in args.target.split(','):
            path = os.path.join(args.data_dir, name + ".bin")
            if os.path.exists(path):
                print(f'Loading dataset: {name} from {path}')
                data_set = load_bin(path, image_size)
                ver_list.append(data_set)
                ver_name_list.append(name)
            else:
                print(f"[WARNING] Dataset file not found: {path}")

        if len(ver_list) == 0:
            raise RuntimeError("No validation datasets loaded. Check --data-dir and --target names.")

        if args.mode == 0:
            if len(nets) == 0:
                raise RuntimeError("No models loaded. Nothing to evaluate.")
            for i, ver_data in enumerate(ver_list):
                dataset_name = ver_name_list[i]
                results = []
                for net in nets:
                    print(f"Running verification for: {dataset_name}")
                    if isinstance(net, mx.mod.Module):
                        raise NotImplementedError("MXNet 'test' function not implemented in this script. Use `dumpR` or adapt code as needed.")
                    else:
                        acc1, std1, acc2, std2, xnorm, embeddings_list, val_metrics = test(
                            ver_data, net, args.batch_size, args.nfolds
                        )
                        print(f"[{dataset_name}] XNorm: {xnorm:.3f}")
                        print(f"[{dataset_name}] No-flip Accuracy: {acc1:.5f} ± {std1:.5f}")
                        print(f"[{dataset_name}] Flip Accuracy: {acc2:.5f} ± {std2:.5f}")
                        results.append(acc2)
                if len(results) > 0:
                    print(f"Max of [{dataset_name}] is {np.max(results):.5f}")
        elif args.mode == 1:
            raise ValueError("Mode 1 not implemented in this script.")
        else:
            if len(nets) == 0:
                raise RuntimeError("No model loaded for dumpR mode.")
            net = nets[0]
            if isinstance(net, mx.mod.Module):
                dumpR(ver_list[0], net, args.batch_size, args.target)
            else:
                raise NotImplementedError("dumpR for PyTorch is not implemented. Use test() or adapt code as needed.")
    except Exception as e:
        print(f"[ERROR] {str(e)}")
