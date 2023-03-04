#!/usr/bin/env python3
import unittest
import random
import sys
import copy
import argparse
import inspect
import collections
import os
import pickle
import gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import torch
import torch.nn as nn
import omniglot

# Import submission
import submission

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#########
# TESTS #
#########

# Baseline
class Test_1b(GradedTestCase):
    def setUp(self):
        pass
     
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_2a(GradedTestCase):
    def setUp(self):
        self.parameters_keys = ['conv0', 'b0', 'conv1', 'b1', 'conv2', 'b2', 'conv3', 'b3', 'w4', 'b4']
        self.submission_maml = submission.MAML(
            num_outputs=5,
            num_inner_steps=1,
            inner_lr=0.4,
            learn_inner_lrs=False,
            outer_lr=0.001,
            log_dir='./logs/'
        )
        self.solution_maml = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.MAML(
            num_outputs=5,
            num_inner_steps=1,
            inner_lr=0.4,
            learn_inner_lrs=False,
            outer_lr=0.001,
            log_dir='./logs/'
        ))
        self.dataloader_train = omniglot.get_omniglot_dataloader(
            split='train',
            batch_size=16,
            num_way=5,
            num_support=1,
            num_query=15,
            num_tasks_per_epoch=240000
        )

    @graded(timeout=5)
    def test_0(self):
        """2a-0-basic: check prediction and accuracies shape for _inner_loop"""
        for i_step, task_batch in enumerate(
                self.dataloader_train,
                start=0
        ):
            for task in task_batch:
                images_support, labels_support, images_query, labels_query = task
                images_support = images_support.to(DEVICE)
                labels_support = labels_support.to(DEVICE)
                images_query = images_query.to(DEVICE)
                labels_query = labels_query.to(DEVICE)
                parameters, accuracies = self.submission_maml._inner_loop(
                    images_support,
                    labels_support,
                    True
                )
                self.assertTrue(parameters['conv0'].shape == torch.Size([64, 1, 3, 3]), "conv0 shape is incorrect")
                self.assertTrue(parameters['b0'].shape == torch.Size([64]), "b0 shape is incorrect")
                self.assertTrue(parameters['conv1'].shape == torch.Size([64, 64, 3, 3]), "conv1 shape is incorrect")
                self.assertTrue(parameters['b1'].shape == torch.Size([64]), "b1 shape is incorrect")
                self.assertTrue(parameters['conv2'].shape == torch.Size([64, 64, 3, 3]), "conv2 shape is incorrect")
                self.assertTrue(parameters['b2'].shape == torch.Size([64]), "b2 shape is incorrect")
                self.assertTrue(parameters['conv3'].shape == torch.Size([64, 64, 3, 3]), "conv3 shape is incorrect")
                self.assertTrue(parameters['b3'].shape == torch.Size([64]), "b3 shape is incorrect")
                self.assertTrue(parameters['w4'].shape == torch.Size([5, 64]), "w4 shape is incorrect")
                self.assertTrue(parameters['b4'].shape == torch.Size([5]), "b4 shape is incorrect")
                self.assertTrue(len(accuracies) == 2, "accuracies length is incorrect")
                break
            break
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_2b(GradedTestCase):
    def setUp(self):
        self.parameters_keys = ['conv0', 'b0', 'conv1', 'b1', 'conv2', 'b2', 'conv3', 'b3', 'w4', 'b4']
        self.submission_maml = submission.MAML(
            num_outputs=5,
            num_inner_steps=1,
            inner_lr=0.4,
            learn_inner_lrs=False,
            outer_lr=0.001,
            log_dir='./logs/'
        )
        self.solution_maml = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.MAML(
            num_outputs=5,
            num_inner_steps=1,
            inner_lr=0.4,
            learn_inner_lrs=False,
            outer_lr=0.001,
            log_dir='./logs/'
        ))
        self.dataloader_train = omniglot.get_omniglot_dataloader(
            split='train',
            batch_size=16,
            num_way=5,
            num_support=1,
            num_query=15,
            num_tasks_per_epoch=240000
        )
    
    @graded(timeout=5)
    def test_0(self):
        """2b-0-basic: check shapes are correct for _outer_step"""
        for i_step, task_batch in enumerate(
                self.dataloader_train,
                start=0
        ):
            self.submission_maml._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query = (
                self.submission_maml._outer_step(task_batch, train=True)
            )
            self.assertTrue(outer_loss.shape == torch.Size([]))
            self.assertTrue(accuracies_support.shape == (2,))
            self.assertTrue(type(accuracy_query) == np.float64)
            break
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
