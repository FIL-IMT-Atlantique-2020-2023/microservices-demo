#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
import traceback
from concurrent import futures
from functools import reduce

import googleclouddebugger
import googlecloudprofiler
from google.auth.exceptions import DefaultCredentialsError
import grpc
from opencensus.ext.stackdriver import trace_exporter as stackdriver_exporter
from opencensus.ext.grpc import server_interceptor
from opencensus.trace import samplers
from opencensus.common.transports.async_ import AsyncTransport

import demo_pb2
import demo_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

import logging
import numpy as np
import pandas as pd
import hashlib
from sklearn.preprocessing import minmax_scale

from reco_utils.common.python_utils import binarize
from reco_utils.common.timer import Timer
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
from reco_utils.recommender.sar import SAR
from reco_utils.recommender.rlrmc.RLRMCalgorithm import RLRMCalgorithm
from reco_utils.recommender.rlrmc.RLRMCdataset import RLRMCdataset
from reco_utils.evaluation.python_evaluation import (
    rmse, mae
)
import sys

from logger import getJSONLogger

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters

# rank of the model, a positive integer (usually small), required parameter
rank_parameter = 10
# regularization parameter multiplied to loss function, a positive number (usually small), required parameter
regularization_parameter = 0.001
# initialization option for the model, 'svd' employs singular value decomposition, optional parameter
initialization_flag = 'svd'  # default is 'random'
# maximum number of iterations for the solver, a positive integer, optional parameter
maximum_iteration = 100  # optional, default is 100
# maximum time in seconds for the solver, a positive integer, optional parameter
maximum_time = 300  # optional, default is 1000

# Verbosity of the intermediate results
verbosity = 0  # optional parameter, valid values are 0,1,2, default is 0
# Whether to compute per iteration train RMSE (and test RMSE, if test data is given)
compute_iter_rmse = True  # optional parameter, boolean value, default is False

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating", "timestamp"]
)


# Convert the float precision to 32-bit in order to reduce memory consumption
data['rating'] = data['rating'].astype(np.float32)

train, test = python_stratified_split(
    data, ratio=0.75, col_user='userID', col_item='itemID', seed=42)

train_users = len(train['userID'].unique())

data = RLRMCdataset(train=train, test=test)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s')

model = RLRMCalgorithm(rank=rank_parameter,
                       C=regularization_parameter,
                       model_param=data.model_param,
                       initialize_flag=initialization_flag,
                       maxiter=maximum_iteration,
                       max_time=maximum_time)

with Timer() as train_time:
    model.fit(data)

print("Took {} seconds for training.".format(train_time.interval))

logger = getJSONLogger('recommendationservice-server')


def initStackdriverProfiling():
    project_id = None
    try:
        project_id = os.environ["GCP_PROJECT_ID"]
    except KeyError:
        # Environment variable not set
        pass

    for retry in range(1, 4):
        try:
            if project_id:
                googlecloudprofiler.start(
                    service='recommendation_server', service_version='1.0.0', verbose=0, project_id=project_id)
            else:
                googlecloudprofiler.start(
                    service='recommendation_server', service_version='1.0.0', verbose=0)
            logger.info("Successfully started Stackdriver Profiler.")
            return
        except (BaseException) as exc:
            logger.info(
                "Unable to start Stackdriver Profiler Python agent. " + str(exc))
            if (retry < 4):
                logger.info(
                    "Sleeping %d seconds to retry Stackdriver Profiler agent initialization" % (retry*10))
                time.sleep(1)
            else:
                logger.warning(
                    "Could not initialize Stackdriver Profiler after retrying, giving up")
    return


class RecommendationService(demo_pb2_grpc.RecommendationServiceServicer):
    def ListRecommendations(self, request, context):
        # max_responses = 5
        # fetch list of products from product catalog stub
        cat_response = product_catalog_stub.ListProducts(demo_pb2.Empty())
        product_ids = [x.id for x in cat_response.products]
        filtered_products = list(set(product_ids)-set(request.product_ids))
        logger.info(request)
        num_products = len(filtered_products)
        # num_return = min(max_responses, num_products)
        # sample list of indicies to return
        user_id = 1 + int(hashlib.sha1(request.user_id.encode("utf-8")
                                       ).hexdigest(), 16) % train_users
        prediction = model.predict(test['userID'], test['itemID'])
        # logger.info(prediction.head())
        # indices = random.sample(range(num_products), num_return)
        # fetch product ids from indices
        # prod_list = [filtered_products[i] for i in indices]
        prod_list = ['1YMWWN1N4O', 'L9ECAV7KIM',
                     'LS4PSXUNUM', '9SIQT8TOJO', 'OLJCESPC7Z']
        logger.info(
            "[Recv ListRecommendations] product_ids={}".format(prod_list))
        # build and return response
        response = demo_pb2.ListRecommendationsResponse()
        response.product_ids.extend(prod_list)
        return response

    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING)

    def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.UNIMPLEMENTED)


if __name__ == "__main__":
    logger.info("initializing recommendationservice")

    try:
        if "DISABLE_PROFILER" in os.environ:
            raise KeyError()
        else:
            logger.info("Profiler enabled.")
            initStackdriverProfiling()
    except KeyError:
        logger.info("Profiler disabled.")

    try:
        if "DISABLE_TRACING" in os.environ:
            raise KeyError()
        else:
            logger.info("Tracing enabled.")
            sampler = samplers.AlwaysOnSampler()
            exporter = stackdriver_exporter.StackdriverExporter(
                project_id=os.environ.get('GCP_PROJECT_ID'),
                transport=AsyncTransport)
            tracer_interceptor = server_interceptor.OpenCensusServerInterceptor(
                sampler, exporter)
    except (KeyError, DefaultCredentialsError):
        logger.info("Tracing disabled.")
        tracer_interceptor = server_interceptor.OpenCensusServerInterceptor()

    try:
        if "DISABLE_DEBUGGER" in os.environ:
            raise KeyError()
        else:
            logger.info("Debugger enabled.")
            try:
                googleclouddebugger.enable(
                    module='recommendationserver',
                    version='1.0.0'
                )
            except (Exception, DefaultCredentialsError):
                logger.error("Could not enable debugger")
                logger.error(traceback.print_exc())
                pass
    except (Exception, DefaultCredentialsError):
        logger.info("Debugger disabled.")

    port = os.environ.get('PORT', "8080")
    catalog_addr = os.environ.get('PRODUCT_CATALOG_SERVICE_ADDR', '')
    if catalog_addr == "":
        raise Exception(
            'PRODUCT_CATALOG_SERVICE_ADDR environment variable not set')
    logger.info("product catalog address: " + catalog_addr)
    channel = grpc.insecure_channel(catalog_addr)
    product_catalog_stub = demo_pb2_grpc.ProductCatalogServiceStub(channel)

    # create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         interceptors=(tracer_interceptor,))

    # add class to gRPC server
    service = RecommendationService()
    demo_pb2_grpc.add_RecommendationServiceServicer_to_server(service, server)
    health_pb2_grpc.add_HealthServicer_to_server(service, server)

    # start server
    logger.info("listening on port: " + port)
    server.add_insecure_port('[::]:'+port)
    server.start()

    # keep alive
    try:
        while True:
            time.sleep(10000)
    except KeyboardInterrupt:
        server.stop(0)
