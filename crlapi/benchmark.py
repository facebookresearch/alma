# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from crlapi import instantiate_class,get_class,get_arguments


class StreamTrainer:
    def create_logger(self, logger_args,all_args):
        self.logger=instantiate_class(logger_args)
        self.logger.save_hps(all_args)

    def create_stream(self, stream_args):
        return instantiate_class(stream_args)

    def create_clmodel(self, cl_model_args):
        from importlib import import_module
        d = dict(cl_model_args)
        if "classname" in d:
            classname = d["classname"]
        else:
            classname = d["class_name"]
        module_path, class_name = classname.rsplit(".", 1)
        module = import_module(module_path)
        c = getattr(module, class_name)
        self.clmodel=c(self.train_stream,cl_model_args)

    def run(self, args):
        self.create_logger(args.logger,args)

        stream_args = args.stream.train
        self.train_stream=self.create_stream(stream_args)

        stream_args = args.stream.evaluation
        self.evaluation_stream=self.create_stream(stream_args)

        clmodel_args = args.clmodel
        self.create_clmodel(clmodel_args)

        evaluation_args = args.evaluation

        #args=_prefix(args,"benchmark/")
        evaluation_mode=evaluation_args.mode
        assert evaluation_mode=="all_tasks" or evaluation_mode=="previous_tasks"

        for n_stage, task in enumerate(self.train_stream):
            self.logger.message("Training at stage "+str(n_stage))
            training_logger = self.logger.get_logger(f"train_stage_{n_stage}/")
            self.clmodel = self.clmodel.update(task, training_logger)
            evaluation_logger = self.logger.get_logger(f"evaluation_stage_{n_stage}/")

            self.logger.message("Evaluation at stage "+str(n_stage))
            for k,evaluation_task in enumerate(self.evaluation_stream):
                if evaluation_mode=="previous_tasks" and k>n_stage:
                    pass
                else:
                    self.logger.message("\tEvaluation on task "+str(k))
                    evaluation=self.clmodel.evaluate(evaluation_task,evaluation_logger,evaluation_args)
                    self.logger.message("\t == "+str(evaluation))
                    for kk,vv in evaluation.items():
                        evaluation_logger.add_scalar(kk,vv,k)

        self.logger.close()
