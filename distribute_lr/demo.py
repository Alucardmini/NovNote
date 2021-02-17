# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/12/26 3:01 PM'

import tensorflow as tf
import sys
import argparse

FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # 从参数服务器和工作主机创建一个集群
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # 创建并启动本地任务的服务器
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # 默认情况下将操作分配给本地Worker
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # 建立模型...
      loss = ...
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # StopAtStepHook 在运行给定步骤后处理停止
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # MonitoredTrainingSession 负责会话初始化
    # 从检查点恢复，保存到检查点，一旦完成或报错就关闭
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # 异步运行训练
        # 有关如何执行同步训练的更多信息，请参见 `tf.train.SyncReplicasOptimizer`
        # mon_sess.run 在被抢占 PS 的情况下处理 AbortedError
        mon_sess.run(train_op)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # 用于定义 tf.train.ClusterSpec 的标志
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
