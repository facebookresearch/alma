#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.utils.tensorboard import SummaryWriter
import sqlite3
import os
import os.path
import csv
import copy
from datetime import datetime
import torch
import numpy as np
import time
import pickle
import bz2
import sys
import pandas as pd
from omegaconf import DictConfig, OmegaConf

class TFPrefixLogger:
    def __init__(self,prefix,logger):
        self.logger=logger
        self.prefix=prefix

    def add_images(self, name, value, iteration):
        self.logger.add_images(self.prefix+name,value,iteration)

    def add_scalar(self, name, value, iteration):
        self.logger.add_scalar(self.prefix+name,value,iteration)

    def add_video(self, name, value, iteration, fps=10):
        self.logger.add_video(self.prefix+name,value,iteration,fps)

    def message(self,msg,from_name=""):
        self.logger.message(msg,from_name=self.prefix+from_name)

    def debug(self,msg,from_name=""):
        self.logger.debug(msg,from_name=self.prefix+from_name)

class TFLogger(SummaryWriter):
    """A logger that stores informations both in tensorboard and CSV formats"""

    def __init__(
        self, log_dir=None,  cache_size=10000, modulo=1,verbose=False, use_zip=True
    ):
        SummaryWriter.__init__(self, log_dir=log_dir)
        self.use_zip = use_zip
        self.save_every = cache_size
        self.modulo=modulo
        self.written_values={}
        self.log_dir = log_dir


        self.verbose = verbose

        self.picklename = log_dir + "/db.pickle.bzip2"
        if not self.use_zip:
            self.picklename = log_dir + "/db.pickle"
        self.to_pickle = []

    def _omegaconf_to_dict(self,hps):
        d={}
        for k,v in hps.items():
            if isinstance(v,DictConfig):
                d[k]=self._omegaconf_to_dict(v)
            else:
                d[k]=v
        return d

    def save_hps(self, hps):
        hps=self._omegaconf_to_dict(hps)
        print(hps)
        f = open(self.log_dir + "/params.json", "wt")
        f.write(str(hps) + "\n")
        f.close()

        outfile = open(self.log_dir + "/params.pickle", "wb")
        pickle.dump(hps, outfile)
        outfile.close()
        self.add_text("Hyperparameters", str(hps))

    def get_logger(self,prefix):
        return TFPrefixLogger(prefix,self)

    def message(self,msg,from_name=""):
        print("[",from_name,"]: ",msg)

    def debug(self,msg,from_name=""):
        print("[DEBUG] [",from_name,"]: ",msg)

    def _to_pickle(self, name, value, iteration):
        self.to_pickle.append((name, iteration, value))
        if len(self.to_pickle) > self.save_every:
            if self.use_zip:
                f = bz2.BZ2File(self.picklename, "ab")
                pickle.dump(self.to_pickle, f)
                f.close()
            else:
                f = open(self.picklename, "ab")
                pickle.dump(self.to_pickle, f)
                f.close()
            self.to_pickle = []

    def add_images(self, name, value, iteration):
        iteration=int(iteration/self.modulo)*self.modulo
        if  (name,iteration) in self.written_values:
            return
        else:
            self.written_values[(name,iteration)]=True


        self._to_pickle(name, value, iteration)
        SummaryWriter.add_images(self, name, value, iteration)

    def add_scalar(self, name, value, iteration):
        iteration=int(iteration/self.modulo)*self.modulo
        if  (name,iteration) in self.written_values:
            return
        else:
            self.written_values[(name,iteration)]=True

        self._to_pickle(name, value, iteration)
        if self.verbose:
            print("['" + name + "' at " + str(iteration) + "] = " + str(value))

        if isinstance(value, int) or isinstance(value, float):
            SummaryWriter.add_scalar(self, name, value, iteration)

    def add_video(self, name, value, iteration, fps=10):
        iteration=int(iteration/self.modulo)*self.modulo
        if  (name,iteration) in self.written_values:
            return
        else:
            self.written_values[(name,iteration)]=True

        self._to_pickle(name, value.numpy(), iteration)
        SummaryWriter.add_video(self, name, value, iteration, fps=fps)

    def close(self):
        if len(self.to_pickle) > 0:
            if self.use_zip:
                f = bz2.BZ2File(self.picklename, "ab")
                pickle.dump(self.to_pickle, f)
                f.close()
            else:
                f = open(self.picklename, "ab")
                pickle.dump(self.to_pickle, f)
                f.close()
            self.to_pickle = []

        SummaryWriter.close(self)

        f = open(self.log_dir + "/done", "wt")
        f.write("Done\n")
        f.close()


class Log:
    def __init__(self, hps, values):
        self.hps = hps
        self.values = values
        max_length = max([len(v) for v in self.values])
        for k in values:
            while len(values[k]) < max_length:
                values[k].append(None)
        self.length = max_length

    def to_xy(self, name):
        assert name in self.values
        x, y = [], []
        for k, v in enumerate(self.values[name]):
            if not v is None:
                x.append(k)
                y.append(v)
        return x, y

    def to_dataframe(self, with_hps=False):
        max_len = np.max([len(k) for v, k in self.values.items()])
        nv = {}
        for k, v in self.values.items():
            while len(v) < max_len:
                v.append(None)
            nv[k] = v
        self.values = nv
        it = [k for k in range(max_len)]
        d = {**self.values, **{"iteration": it}}

        if with_hps:
            for k in self.hps:
                dd = [self.hps[k] for i in range(max_len)]
                d = {**d, **{"_hp/" + k: dd}}

        return pd.DataFrame(d)

    def get_at(self, name, iteration):
        return self.values[name][iteration]

    def get(self, name, keep_none=False):
        v = self.values[name]
        if not keep_none:
            return [k for k in v if not k is None]
        else:
            return v

    def replace_None_(self, name):
        v = self.values[name]
        last_v = None
        first_v = None
        r = []
        for k in range(len(v)):
            if v[k] is None:
                r.append(last_v)
            else:
                r.append(v[k])
                if last_v is None:
                    first_v = v[k]
                last_v = v[k]

        p = 0
        while r[p] is None:
            r[p] = first_v
            p += 1
        self.values[name] = r

    def max(self, name):
        v = self.values[name]
        vv = [k for k in v if not k is None]
        return np.max(vv)

    def min(self, name):
        v = self.values[name]
        vv = [k for k in v if not k is None]
        return np.min(vv)

    def argmin(self, name):
        v = self.values[name]
        vv = [k for k in v if not k is None]
        _max = np.max(vv)

        for k in range(len(v)):
            if v[k] is None:
                vv.append(_max + 1.0)
            else:
                vv.append(v[k])
        return np.argmin(vv)

    def argmax(self, name):
        v = self.values[name]
        vv = [k for k in v if not k is None]
        _min = np.min(vv)
        vv = []
        for k in range(len(v)):
            if v[k] is None:
                vv.append(_min - 1.0)
            else:
                vv.append(v[k])
        return np.argmax(vv)


class Logs:
    def __init__(self):
        self.logs = []
        self.hp_names = None
        self.filenames = []

    def _add(self, log):
        self.hp_names = {k: True for k in log.hps}
        for l in self.logs:
            for k in log.hps:
                if not k in l.hps:
                    l.hps[k] = "none"

        self.logs.append(log)

    def add(self, logs):
        if isinstance(logs, Log):
            self._add(logs)
        else:
            for l in logs:
                self._add(l)

    def max(self, function):
        alls = [function(l) for l in self.logs]
        idx = np.argmax(alls)
        return self.logs[idx]

    def hps(self):
        return list(self.hp_names)

    def size(self):
        return len(self.logs)

    def filter(self, hp_name, test_fn):
        logs = Logs()
        if not callable(test_fn):
            for l in self.logs:
                h = l.hps[hp_name]
                if h == test_fn:
                    logs.add(l)
        else:
            for l in self.logs:
                if test_fn(l.hps[hp_name]):
                    logs.add(l)
        return logs

    def unique_hps(self, name):
        r = {}
        for l in self.logs:
            v = l.hps[name]
            r[v] = 1
        return list(r.keys())

    def __len__(self):
        return len(self.logs)

    def to_dataframe(self):
        rdf = None
        for log in self.logs:
            df = log.to_dataframe(with_hps=True)
            if rdf is None:
                rdf = df
            else:
                rdf = pd.concat([rdf, df])
        return rdf

    # def plot(self, y, x, hue=None, style=None, row=None, col=None, kind="line"):


def flattify(d):
    d=dict(d)
    r = {}
    for k, v in d.items():
        if isinstance(v, dict):
            rr = flattify(v)
            rrr = {k + "/" + kk: rr[kk] for kk in rr}
            r = {**r, **rrr}
        elif isinstance(v, list):
            r[k] = str(v)
        else:
            r[k] = v
    return r


def read_log(directory, use_bz2=True, debug=False):
    print("== Read ", directory)
    # if os.path.exists(directory+"/fast.pickle"):
    #     f=open(directory+"/fast.pickle","rb")
    #     log=pickle.load(f)
    #     f.close()
    #     return log

    f = None
    if use_bz2:
        picklename = directory + "/db.pickle.bzip2"
        f = bz2.BZ2File(picklename, "rb")
    else:
        picklename = directory + "/db.pickle"
        f = open(picklename, "rb")
    values = {}

    try:
        while True:
            a = pickle.load(f)
            if not a is None:
                for name, iteration, value in a:
                    # print(name,iteration,value)
                    if debug:
                        print(name, value, type(value))
                    if isinstance(value, np.int64):
                        value = int(value)
                    if (
                        isinstance(value, int)
                        or isinstance(value, float)
                        or isinstance(value, str)
                    ):
                        if not name in values:
                            values[name] = []
                        while len(values[name]) < iteration + 1:
                            values[name].append(None)
                        values[name][iteration] = value
    except:
        f.close()

    f = open(directory + "/params.pickle", "rb")
    params = pickle.load(f)
    params = flattify(params)
    f.close()
    log = Log(params, values)
    log.from_directory = directory
    # f=open(directory+"/fast.pickle","wb")
    # pickle.dump(log,f)
    # f.close()

    return log


def read_directory(directory, use_bz2=True):
    import os
    import os.path

    l = Logs()
    name = "db.pickle"
    if use_bz2:
        name = "db.pickle.bzip2"
    for dirpath, dirnames, filenames in os.walk(directory):
        if name in filenames:
            log = read_log(dirpath, use_bz2)
            l.add(log)
    print("Found %d logs" % l.size())
    return l


def _create_col(df, hps, _name):
    vs = []
    for k, v in df.groupby(hps):
        n = {hps[i]: k[i] for i in range(len(hps))}
        v = v.copy()
        name = ",".join([str(k) + "=" + str(n[k]) for k in n])
        print(name)
        print(_name)
        v[_name] = name
        vs.append(v)
    return pd.concat(vs)


def plot_dataframe(
    df, y, x="iteration", hue=None, style=None, row=None, col=None, kind="line"
):
    import seaborn as sns

    cols = [y, x]
    if isinstance(row, list):
        cols += row
    else:
        cols += [row]
    if isinstance(col, list):
        cols += col
    else:
        cols += [col]
    if isinstance(style, list):
        cols += style
    else:
        cols += [style]
    if isinstance(hue, list):
        cols += hue
    else:
        cols += [hue]
    cols = [c for c in cols if not c is None]
    df = df[cols].dropna()

    if isinstance(row, list):
        df = _create_col(df, row, "__row")
        row = "__row"
    if isinstance(col, list):
        df = _create_col(df, col, "__col")
        col = "__col"
    if isinstance(style, list):
        df = _create_col(df, style, "__style")
        style = "__style"
    if isinstance(hue, list):
        df = _create_col(df, hue, "__hue")
        hue = "__hue"

    # df = convert_iteration_to_steps(df)

    sns.relplot(x=x, y=y, hue=hue, style=style, row=row, col=col, data=df, kind=kind)
