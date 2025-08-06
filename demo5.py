# fedavg from https://github.com/WHDY/FedAvg/
# TODO redistribute offline() based on very transaction, not at the beginning of every loop
# TODO when accepting transactions, check comm_round must be in the same, that is to assume by default they only accept the transactions that are in the same round, so the final block contains only updates from the same round
# TODO subnets - potentially resolved by let each miner sends block
# TODO let's not remove peers because of offline as peers may go back online later, instead just check if they are online or offline. Only remove peers if they are malicious. In real distributed network, remove peers if cannot reach for a long period of time
# assume no resending transaction mechanism if a transaction is lost due to offline or time out. most of the time, unnecessary because workers always send the newer updates, if it's not the last worker's updates
# assume just skip verifying a transaction if offline, in reality it may continue to verify what's left
# PoS also uses resync chain - the chain with highter stake
# only focus on catch malicious worker
# TODO need to make changes in these functions on Sunday
# pow_resync_chain
# update_model_after_chain_resync
# TODO miner sometimes receives worker transactions directly for unknown reason - discard tx if it's not the correct type
# TODO a chain is invalid if a malicious block is identified after this miner is identified as malicious
# TODO Do not associate with blacklisted node. This may be done already.
# TODO KickR continuousness should skip the rounds when nodes are not selected as workers
# TODO update forking log after loading network snapshots
# TODO in reuqest_to_download, forgot to check for maliciousness of the block miner
# future work
# TODO - non-even dataset distribution
import os

os.environ["KMP_DUPLICATE_LIB_O"] = "TRUE"

import os
import sys
import argparse
import numpy as np
import random
import time
from datetime import datetime
import copy
from sys import getsizeof
import sqlite3
import pickle
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from Models import Mnist_2NN, Mnist_CNN
from Device import Device, DevicesInNetwork
from Block import Block
from Blockchain import Blockchain

# set program execution time for logging purpose
date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
log_files_folder_path = f"logs/{date_time}"
NETWORK_SNAPSHOTS_BASE_FOLDER = "snapshots"
# for running on Google Colab
# log_files_folder_path = f"/content/drive/MyDrive/BFA/logs/{date_time}"
# NETWORK_SNAPSHOTS_BASE_FOLDER = "/content/drive/MyDrive/BFA/snapshots"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Block_FedAvg_Simulation")

# debug attributes
# debug 属性
parser.add_argument('-g', '--gpu', type=str, default='all-MiniLM-L6-v2', help='gpu id to use(e.g. 0,all-MiniLM-L6-v2,2,3)')
parser.add_argument('-v', '--verbose', type=int, default=1, help='打印详细调试日志')
parser.add_argument('-sn', '--save_network_snapshots', type=int, default=0,
                    help='如果设置为1，只保存network_snapshots; 将在快照文件夹中创建一个带有日期的文件夹')
parser.add_argument('-dtx', '--destroy_tx_in_block', type=int, default=0,
                    help='目前，存储在块中的事务占用GPU内存，并且还没有找到将它们移动到CPU内存或硬盘的方法, 所以打开它来节省GPU内存，以便PoS运行100+轮。如果需要执行链重同步，则不太好。')
parser.add_argument('-rp', '--resume_path', type=str, default=None, help='从保存network_snapshots的路径恢复；只提供日期')
parser.add_argument('-sf', '--save_freq', type=int, default=5, help='network_snapshot的保存频率')
parser.add_argument('-sm', '--save_most_recent', type=int, default=2,
                    help='为了节省空间，只保留最近指定数量的快照；0表示保留全部')

# FL attributes
# 联邦学习属性
parser.add_argument('-B', '--batchsize', type=int, default=100, help='本地训练尺寸')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='训练的模型')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01,
                    help="学习率, use value from origin paper as default")
parser.add_argument('-op', '--optimizer', type=str, default="SGD", help='要使用的优化器，默认情况下实现随机梯度下降')
parser.add_argument('-iid', '--IID', type=int, default=1, help='将数据分配给设备的方式')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=20, help='最大通信轮数，如果收敛可能提前终止')
parser.add_argument('-nd', '--num_devices', type=int, default=20, help='模拟网络中设备的数量')
parser.add_argument('-st', '--shard_test_data', type=int, default=1,
                    help='当测试数据集没有分片时，很容易看到全局模型在设备之间是一致的')
parser.add_argument('-nm', '--num_malicious', type=int, default=0,
                    help="网络中恶意节点的数量。恶意节点的数据集将引入高斯噪声")
parser.add_argument('-nv', '--noise_variance', type=int, default=0, help="注入高斯噪声的噪声方差水平")
parser.add_argument('-le', '--default_local_epochs', type=int, default=5,
                    help='本地训练. 如果未指定-mt，则对每个worker按相同的epoch数训练本地模型')

# blockchain system consensus attributes
# 区块链系统共识属性

parser.add_argument('-ur', '--unit_reward', type=int, default=1,
                    help='一系列的奖励unit reward for providing data, verification of signature, validation and so forth')
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=6,
                    help="如果在此轮数中被识别为恶意设备，则工作器或验证器设备将被踢出设备的对等列表（放入黑名单）")
parser.add_argument('-lo', '--lazy_worker_knock_out_rounds', type=int, default=10,
                    help="如果一个工作设备没有为这个轮数提供更新，那么它就会被踢出设备的对等体列表（放入黑名单），因为它太慢或懒得进行更新，只接受模型更新。（不要在意懒惰的验证者或矿工，因为他们不会得到奖励）")
parser.add_argument('-pow', '--pow_difficulty', type=int, default=0, help="如果设置为0，则表示矿工正在使用PoS")

# blockchain FL miner/miner restriction tuning parameters
# 区块链FL验证器/矿工限制调优参数
parser.add_argument('-mt', '--miner_acception_wait_time', type=float, default=0.0,
                    help="矿工接受交易的默认时间窗口，以秒为单位。0表示没有时间限制，每个设备每轮只执行相同数量（-le）的epoch，就像fedag论文中一样")
parser.add_argument('-ml', '--miner_accepted_transactions_size_limit', type=float, default=0.0,
                    help="在此限制之后，矿工将不再接受任何交易。0表示没有大小限制。必须指定this或-mt，或者两者都指定。这个参数决定了最终的block_size")
parser.add_argument('-mp', '--miner_pos_propagated_block_wait_time', type=float, default=float("inf"),
                    help="此等待时间从通信轮开始计算，用于模拟PoS中的分叉事件")
parser.add_argument('-vh', '--miner_threshold', type=float, default=1.0, help="一个准确度的阈值来判断恶意工作者")
parser.add_argument('-md', '--malicious_updates_discount', type=float, default=0.0,
                    help="do not entirely drop the voted negative worker transaction because that risks the same worker dropping the entire transactions and repeat its accuracy again and again and will be kicked out. Apply a discount factor instead to the false negative worker's updates are by some rate applied so it won't repeat")
parser.add_argument('-mv', '--malicious_miner_on', type=int, default=0, help="让恶意验证器翻转投票结果")

# distributed system attributes
# 分布式系统属性
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='设备在线的几率')
parser.add_argument('-els', '--even_link_speed_strength', type=int, default=1,
                    help="该变量用于模拟传输延迟。缺省值1表示为每个设备分配相同的链路速度强度-dts字节/秒。如果设置为0，则链路速度强度在0到1之间随机启动，这意味着设备在实验期间将传输-els*-dts字节/秒，一次交易约为35k字节。")
parser.add_argument('-dts', '--base_data_transmission_speed', type=float, default=70000.0,
                    help="当-els == 1时，每秒可传输的数据量。设置这个变量来决定传输速度（带宽），这进一步决定了传输延迟——在实验中，一个事务大约是35k字节。")
parser.add_argument('-ecp', '--even_computation_power', type=int, default=1,
                    help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value all-MiniLM-L6-v2 means evenly assign computation power to all-MiniLM-L6-v2. If set to 0, power is randomly initiated as an int between 0 and 4, both included.")

# simulation attributes
# 仿真属性
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*',
                    help="hard assign number of roles in the network, order by worker, miner and miner. e.g. 12,5,3 assign 12 workers, 5 miners and 3 miners. \"*,*,*\" means completely random role-assigning in each communication round ")
parser.add_argument('-aio', '--all_in_one', type=int, default=1, help='让所有节点在注册时都知道网络中的彼此')
parser.add_argument('-cs', '--check_signature', type=int, default=1,
                    help='如果设置为0，则假定所有签名都经过验证，以节省执行时间')

# parser.add_argument('-la', '--least_assign', type=str, default='*,*,*', help='the assigned number of roles are at least guaranteed in the network')

if __name__ == "__main__":

    # create logs/ if not exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # get arguments
    args = parser.parse_args()
    args = args.__dict__

    # detect CUDA
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # pre-define system variables
    latest_round_num = 0

    ''' If network_snapshot is specified, continue from left '''
    if args['resume_path']:
        if not args['save_network_snapshots']:
            print("注意：save_network_snapshots设置为0。继续将不会保存新的network_snapshots")
        network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{args['resume_path']}"
        latest_network_snapshot_file_name = \
        sorted([f for f in os.listdir(network_snapshot_save_path) if not f.startswith('.')],
               key=lambda fn: int(fn.split('_')[-1]), reverse=True)[0]
        print(f"Loading network snapshot from {args['resume_path']}/{latest_network_snapshot_file_name}")
        print("注意：加载的dev env必须与当前的dev env相同，即cpu、gpu或gpu并行")
        latest_round_num = int(latest_network_snapshot_file_name.split('_')[-1])
        devices_in_network = pickle.load(
            open(f"{network_snapshot_save_path}/{latest_network_snapshot_file_name}", "rb"))
        devices_list = list(devices_in_network.devices_set.values())
        log_files_folder_path = f"logs/{args['resume_path']}"
        # for colab
        # log_files_folder_path = f"/content/drive/MyDrive/BFA/logs/{args['resume_path']}"
        # original arguments file
        args_used_file = f"{log_files_folder_path}/args_used.txt"
        file = open(args_used_file, "r")
        log_whole_text = file.read()
        lines_list = log_whole_text.split("\n")
        for line in lines_list:
            # abide by the original specified rewards
            if line.startswith('--unit_reward'):
                rewards = int(line.split(" ")[-1])
            # get number of roles
            if line.startswith('--hard_assign'):
                roles_requirement = line.split(" ")[-1].split(',')
            # get mining consensus
            if line.startswith('--pow_difficulty'):
                mining_consensus = 'PoW' if int(line.split(" ")[-1]) else 'PoS'
        # determine roles to assign
        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1

        try:
            miners_needed = int(roles_requirement[1])
        except:
            miners_needed = 1
    else:
        ''' 从头开始设置'''

        # 0. 如果不是按照快照开始，则创建log_files_folder_path
        os.mkdir(log_files_folder_path)

        # all-MiniLM-L6-v2. save arguments used（存使用的参数）
        with open(f'{log_files_folder_path}/args_used.txt', 'w') as f:
            f.write("Command line arguments used：使用的命令行参数 -\n")
            f.write(' '.join(sys.argv[1:]))
            f.write("\n\nAll arguments used -\n")
            for arg_name, arg in args.items():
                f.write(f'\n--{arg_name} {arg}')

        # 2. create network_snapshot folder（创建网络快照文件夹）
        if args['save_network_snapshots']:
            network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{date_time}"
            os.mkdir(network_snapshot_save_path)

        # 3. assign system variables（分配系统变量）
        # for demonstration purposes, this reward is for every rewarded action
        rewards = args["unit_reward"]

        # 4. get number of roles needed in the network（从参数中得到各个人员分配数）
        roles_requirement = args['hard_assign'].split(',')
        # determine roles to assign
        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1

        try:
            miners_needed = int(roles_requirement[1])
        except:
            miners_needed = 1

        # 5. check arguments eligibility（检查参数合格性）

        num_devices = args['num_devices']
        num_malicious = args['num_malicious']

        if num_devices < workers_needed + miners_needed :#+ miners_needed:
            print(workers_needed)
            print(miners_needed)
            sys.exit("ERROR: 分配给设备的角色超过了网络中允许的最大设备数量.")

        if num_devices < 2:
            sys.exit(
                "ERROR: 网络中设备不足.\n The system needs at least one miner, one worker and/or one miner to start the operation.\nSystem aborted.")

        net = None
        if args['model_name'] == 'mnist_2nn':
            net = Mnist_2NN()
        elif args['model_name'] == 'mnist_cnn':
            net = Mnist_CNN()

        # 7. 如果网络可用，则分配GPU，否则分配CPU
        # os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        print(f"{torch.cuda.device_count()} GPUs are available to use!")
        net = net.to(dev)

        # 8. set loss_function（设置损失函数）
        loss_func = F.cross_entropy

        # 9. create devices in the network
        devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=args['IID'], batch_size=args['batchsize'],
                                              learning_rate=args['learning_rate'], loss_func=loss_func,
                                              opti=args['optimizer'], num_devices=num_devices,
                                              network_stability=args['network_stability'], net=net, dev=dev,
                                              knock_out_rounds=args['knock_out_rounds'],
                                              lazy_worker_knock_out_rounds=args['lazy_worker_knock_out_rounds'],
                                              shard_test_data=args['shard_test_data'],
                                              miner_acception_wait_time=args['miner_acception_wait_time'],
                                              miner_accepted_transactions_size_limit=args[
                                                  'miner_accepted_transactions_size_limit'],
                                              miner_threshold=args['miner_threshold'],
                                              pow_difficulty=args['pow_difficulty'],
                                              even_link_speed_strength=args['even_link_speed_strength'],
                                              base_data_transmission_speed=args['base_data_transmission_speed'],
                                              even_computation_power=args['even_computation_power'],
                                              malicious_updates_discount=args['malicious_updates_discount'],
                                              num_malicious=num_malicious, noise_variance=args['noise_variance'],
                                              check_signature=args['check_signature'],
                                              not_resync_chain=args['destroy_tx_in_block'])
        del net
        devices_list = list(devices_in_network.devices_set.values())
        print(devices_list)

        # 10. register devices and initialize global parameterms(注册设备和初始化全局参数)
        for device in devices_list:
            # set initial global weights(设置初始全局权重)
            device.init_global_parameters()
            # helper function for registration simulation - set devices_list and aio（注册模拟的辅助功能-设置devices_list和aio）
            device.set_devices_dict_and_aio(devices_in_network.devices_set, args["all_in_one"])
            # simulate peer registration, with respect to device idx order（模拟对等注册，相对于设备idx顺序）
            device.register_in_the_network()
        # remove its own from peer list if there is(如果有的话，从对等列表中删除它自己)
        for device in devices_list:
            device.remove_peers(device)

        # 11. build logging files/database path（构建日志文件/数据库路径）
        # create log files
        # 构建日志文件/数据库路径
        open(f"{log_files_folder_path}/correctly_kicked_workers.txt", 'w').close()
        open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'w').close()
        open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt", 'w').close()
        open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'w').close()
        # open(f"{log_files_folder_path}/correctly_kicked_miners.txt", 'w').close()
        # open(f"{log_files_folder_path}/mistakenly_kicked_miners.txt", 'w').close()
        open(f"{log_files_folder_path}/kicked_lazy_workers.txt", 'w').close()

        # 12. set_up the mining consensus
        mining_consensus = 'PoW' if args['pow_difficulty'] else 'PoS'

    # create malicious worker identification database(创建恶意工作人员识别数据库)
    conn = sqlite3.connect(f'{log_files_folder_path}/malicious_wokrer_identifying_log.db')
    conn_cursor = conn.cursor()
    conn_cursor.execute("""CREATE TABLE if not exists  malicious_workers_log (
	device_seq text,
	if_malicious integer,
	correctly_identified_by text,
	incorrectly_identified_by text,
	in_round integer,
	when_resyncing text
	)""")

    # VBFL starts here
    for comm_round in range(latest_round_num + 1, args['max_num_comm'] + 1):
        # create round specific log folder
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if os.path.exists(log_files_folder_path_comm_round):
            print(f"Deleting {log_files_folder_path_comm_round} and create a new one.")
            shutil.rmtree(log_files_folder_path_comm_round)
        os.mkdir(log_files_folder_path_comm_round)
        # free cuda memory
        if dev == torch.device("cuda"):
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()
        print(f"\nCommunication round {comm_round}")
        comm_round_start_time = time.time()
        # (RE)ASSIGN ROLES
        workers_to_assign = workers_needed
        miners_to_assign = miners_needed
        workers_this_round = []
        miners_this_round = []
        random.shuffle(devices_list)
        for device in devices_list:
            if workers_to_assign:
                device.assign_worker_role()
                workers_to_assign -= 1
            elif miners_to_assign:
                device.assign_miner_role()
                miners_to_assign -= 1
            else:
                device.assign_role()
            # 随机分配
            if device.return_role() == 'worker':
                workers_this_round.append(device)
            elif device.return_role() == 'miner':
                miners_this_round.append(device)
            device.online_switcher()
        # 分配角色，设置在线与否


        ''' DEBUGGING CODE '''
        if args['verbose']:

            # show devices initial chain length and if online
            # 显示设备的初始链长度和是否在线
            for device in devices_list:
                if device.is_online():
                    print(f'{device.return_idx()} {device.return_role()} online - ', end='')
                else:
                    print(f'{device.return_idx()} {device.return_role()} offline - ', end='')
                # debug chain length
                print(f"区块链长度为 {device.return_blockchain_object().return_chain_length()}")

            # show device roles
            print(
                f"\n在本轮训练中 {len(workers_this_round)}名工人, {len(miners_this_round)} 名矿工 .")
            print("\n这一轮的工人是")
            for worker in workers_this_round:
                print(f"d_{worker.return_idx().split('_')[-1]} online - {worker.is_online()} with chain len {worker.return_blockchain_object().return_chain_length()}")
            print("\n这一轮的矿工是")
            for miner in miners_this_round:
                print(f"d_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_blockchain_object().return_chain_length()}")
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
            for device_seq, device in devices_in_network.devices_set.items():
                peers = device.return_peers()
                print(f"d_{device_seq.split('_')[-1]} - {device.return_role()[0]} has peer list ", end='')
                for peer in peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")

        ''' DEBUGGING CODE ENDS '''

        # re-init round vars - in real distributed system, they could still fall behind in comm round, but here we assume they will all go into the next round together, thought device may go offline somewhere in the previous round and their variables were not therefore reset
        # 重新初始化轮变量——在真实的分布式系统中，它们仍然可能在通信轮中落后，但这里我们假设它们都将一起进入下一轮，因为设备可能在上一轮的某个地方离线，因此它们的变量不会被重置
        for miner in miners_this_round:
            miner.miner_reset_vars_for_new_round()
        for worker in workers_this_round:
            worker.worker_reset_vars_for_new_round()
        # DOESN'T MATTER ANY MORE AFTER TRACKING TIME, but let's keep it - orginal purpose: shuffle the list(for worker, this will affect the order of dataset portions to be trained)
        # 跟踪时间结束后不再重要，但让我们保留它 - 最初的目的：洗牌列表（对于worker，这将影响要训练的数据集部分的顺序）
        random.shuffle(workers_this_round)
        random.shuffle(miners_this_round)

        ''' 工人、验证者和矿工轮流完成工作 '''

        print('''---------步骤1 -工人分配相关的矿工(并进行本地更新，但它是在步骤2的代码块中实现的) \n''')
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
            # 重新同步链（由于上一轮的分叉，区块可能会被丢弃）
            if worker.resync_chain(mining_consensus):
                worker.update_model_after_chain_resync(log_files_folder_path_comm_round, conn, conn_cursor)
            # Worker（应该）执行本地更新和关联
            print(f"{worker.return_idx()} - worker {worker_iter + 1}/{len(workers_this_round)} 工人节点将尝试与矿工建立关联，以便进行后续的交易和更新传递...")#修改
            # 工人与矿工合作接受最终开采的区块
            if worker.online_switcher():
                associated_miner = worker.associate_with_device("miner")
                if associated_miner:
                    associated_miner.add_device_to_association(worker)
                    print(f"与该矿工关联：{associated_miner.return_idx()}")
                else:
                    print(f"没找到一个合适的矿工节点（miner）为了id为{worker.return_idx()}工作节点 peer list.")
        print(
            ''' ---------步骤2 -矿工接受本地更新，并将其广播到各自对等列表中的其他矿工（在此步骤中调用工作器local_updates()）。\n''')
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if miner.resync_chain(mining_consensus):
                miner.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
            associated_workers = list(miner.return_associated_workers())
            if not associated_workers:
                print(f"在这一矿工沟通中，没有找到相关联的工作者 {miner.return_idx()} {miner_iter + 1}/{len(miners_this_round)} .")
                continue
            miner_link_speed = miner.return_link_speed()
            print(f"如果在线的话，{miner.return_idx()} - 矿工 {miner_iter + 1}/{len(miners_this_round)} 是否接受工人的更新与链接速度{miner_link_speed} 字节/秒")
            # Records_dict用于记录每个epoch的传输延迟，以确定下一个epoch更新的到达时间
            records_dict = dict.fromkeys(associated_workers, None)
            for worker, _ in records_dict.items():
                records_dict[worker] = {}
            # 用于到达时间的简单排序，以便稍后的验证器广播（和矿工的接受顺序）
            transaction_arrival_queue = {}
            # 这里调用的# workers local_updates（）作为他们的更新传输可能受到矿工接受时间和/或大小的限制
            if "矿工没有指定等待时间"=="矿工没有指定等待时间":
                # 没有指定等待时间。每个相关的工作执行指定数量的局部epoch
                for worker_iter in range(len(associated_workers)):
                    worker = associated_workers[worker_iter]
                    if "该worker不是黑名单" == "该worker不是黑名单":
                        print(f'worker {worker_iter + 1}/{len(associated_workers)} of miner {miner.return_idx()} 正在进行本地更新')
                        if worker.online_switcher():
                            local_update_spent_time = worker.worker_local_update(rewards,log_files_folder_path_comm_round,comm_round, local_epochs=args['default_local_epochs'])
                            worker_link_speed = worker.return_link_speed()
                            lower_link_speed = miner_link_speed if miner_link_speed < worker_link_speed else worker_link_speed
                            # 整个架构中传输的信息⬇
                            unverified_transaction = worker.return_local_updates_and_signature(comm_round)
                            print("worker首次训练出来的事务key值")
                            print(unverified_transaction.keys())
                            unverified_transactions_size = getsizeof(str(unverified_transaction))
                            transmission_delay = unverified_transactions_size / lower_link_speed
                            if miner.online_switcher():
                                transaction_arrival_queue[local_update_spent_time + transmission_delay] = unverified_transaction
                                print(f"矿工 {miner.return_idx()} 已经接收到事务.")
                            else:
                                print(
                                    f"矿工 {miner.return_idx()} 下线了，没法接收事务")
                        else:
                            print(f"劳工 {worker.return_idx()} 下线了，没法本地更新")
                    else:
                        print(
                            f"worker {worker.return_idx()} in miner {miner.return_idx()}'s black list. This worker's transactions won't be accpeted.")
            miner.set_unordered_arrival_time_accepted_worker_transactions(transaction_arrival_queue)
            # 以防验证器脱机接受广播的事务，但稍后可以重新联机以验证自己接收的事务
            miner.set_transaction_for_final_validating_queue(sorted(transaction_arrival_queue.items()))
            # broadcast to other miners
            if transaction_arrival_queue:
                miner.miner_broadcast_worker_transactions()
                print("接收到的事务，已经广播到其他矿工了")
            else:
                print("这个矿工没有接收到任何事务，可能是由于工作器和/或矿工脱机或在执行本地更新或传输更新时超时，或者所有工作器都在验证器的黑名单中。")

        print('''---------步骤2.5——通过广播的工作者事务，矿工决定最终的事务到达顺序 \n''')
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            accepted_broadcasted_miner_transactions = miner.return_accepted_broadcasted_worker_transactions()
            print(f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} 正在通过组合接收到的直接工作者事务和接收到的广播事务来计算最终事务到达顺序…")
            accepted_broadcasted_transactions_arrival_queue = {}
            if accepted_broadcasted_miner_transactions:
                print("成功接受到其他矿工的广播序列")
                # 计算广播的事务到达时间
                self_miner_link_speed = miner.return_link_speed()
                for broadcasting_miner_record in accepted_broadcasted_miner_transactions:
                    broadcasting_miner_link_speed = broadcasting_miner_record['source_miner_link_speed']
                    lower_link_speed = self_miner_link_speed if self_miner_link_speed < broadcasting_miner_link_speed else broadcasting_miner_link_speed
                    for arrival_time_at_broadcasting_miner, broadcasted_transaction in broadcasting_miner_record['broadcasted_transactions'].items():
                        print("传播后的事务key值")
                        print(broadcasted_transaction.keys())
                        transmission_delay = getsizeof(str(broadcasted_transaction)) / lower_link_speed
                        accepted_broadcasted_transactions_arrival_queue[transmission_delay + arrival_time_at_broadcasting_miner] = broadcasted_transaction
            else:
                print(
                    f"miner {miner.return_idx()} {miner_iter + 1}/{len(miners_this_round)} 没有收到任何广播工作者事务。")
            # 将board - cast事务与直接接受的事务混合
            final_transactions_arrival_queue = sorted(
                {**miner.return_unordered_arrival_time_accepted_worker_transactions(),**accepted_broadcasted_transactions_arrival_queue}.items())
            miner.set_transaction_for_final_validating_queue(final_transactions_arrival_queue)
            print(f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} 完成了已排序的最终事务到达顺序的计算. 总共接受到的序列长度是{len(final_transactions_arrival_queue)}条")

        print(''' -----------Step 3 - 矿工按照事务到达时间的顺序进行自我验证和交叉验证（验证来自工作器的本地更新）.\n''')
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            final_transactions_arrival_queue = miner.return_final_transactions_validating_queue()
            if final_transactions_arrival_queue:
                print("小小矿工接收到了整理好的序列了哦，开始验证！")
                # 验证器异步地在它自己的测试集上执行一次更新和验证
                local_validation_time = miner.miner_update_model_by_one_epoch_and_validate_local_accuracy(args['optimizer'])
                print(f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} 验证器按照事务到达时间的顺序进行自我验证和交叉验证")
                for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
                    print("首次验证前的事务key值")
                    print(unconfirmmed_transaction.keys())
                    if miner.online_switcher():
                        # 验证将不会开始，直到验证器本地完成更新和验证的一个epoch（工作事务将排队）。
                        if arrival_time < local_validation_time:
                            arrival_time = local_validation_time
                        # 验证的手段
                        validation_time, valid_miner_sig_candidate_transacitons, is_miner_sig_valid = miner.verify_miner_transaction1(unconfirmmed_transaction, rewards, log_files_folder_path, comm_round,args['malicious_miner_on'])#验证的手段

                        begin_mining_time = 0
                        new_begin_mining_time = begin_mining_time
                        valid_miner_sig_candidate_transacitons = []
                        invalid_miner_sig_candidate_transacitons = []
                        if "hello" == "hello":
                            if is_miner_sig_valid:
                                miner_info_this_tx = {
                                    'miner': unconfirmmed_transaction['validation_done_by'],
                                    'validation_rewards': unconfirmmed_transaction['validation_rewards'],
                                    'validation_time': unconfirmmed_transaction['validation_time'],
                                    'miner_rsa_pub_key': unconfirmmed_transaction['miner_rsa_pub_key'],
                                    'miner_signature': unconfirmmed_transaction['miner_signature'],
                                    'update_direction': unconfirmmed_transaction['update_direction'],
                                    'miner_device_idx': miner.return_idx(),
                                    'miner_verification_time': validation_time,
                                    'miner_rewards_for_this_tx': rewards}
                                # 验证器的事务签名有效
                                found_same_worker_transaction = False
                                for valid_miner_sig_candidate_transaciton in valid_miner_sig_candidate_transacitons:
                                    if valid_miner_sig_candidate_transaciton['worker_signature'] == unconfirmmed_transaction['worker_signature']:
                                        found_same_worker_transaction = True
                                        print("交易重复")
                                        break
                                if not found_same_worker_transaction:
                                    valid_miner_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
                                    del valid_miner_sig_candidate_transaciton['validation_done_by']
                                    del valid_miner_sig_candidate_transaciton['validation_rewards']
                                    del valid_miner_sig_candidate_transaciton['update_direction']
                                    del valid_miner_sig_candidate_transaciton['validation_time']
                                    del valid_miner_sig_candidate_transaciton['miner_rsa_pub_key']
                                    del valid_miner_sig_candidate_transaciton['miner_signature']
                                    valid_miner_sig_candidate_transaciton['positive_direction_miners'] = []
                                    valid_miner_sig_candidate_transaciton['negative_direction_miners'] = []
                                    valid_miner_sig_candidate_transacitons.append(
                                        valid_miner_sig_candidate_transaciton)
                                    print("有效事务:")
                                if unconfirmmed_transaction['update_direction']:
                                    valid_miner_sig_candidate_transaciton['positive_direction_miners'].append(miner_info_this_tx)
                                else:
                                    valid_miner_sig_candidate_transaciton['negative_direction_miners'].append(miner_info_this_tx)
                                transaction_to_sign = valid_miner_sig_candidate_transaciton
                            else:
                                print("无效签名")
                                # miner's transaction signature invalid
                                invalid_miner_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
                                invalid_miner_sig_candidate_transaciton['miner_verification_time'] = verification_time
                                invalid_miner_sig_candidate_transaciton['miner_rewards_for_this_tx'] = rewards
                                invalid_miner_sig_candidate_transacitons.append(invalid_miner_sig_candidate_transaciton)
                                print("无效事务")
                                print(invalid_miner_sig_candidate_transacitons)
                                transaction_to_sign = invalid_miner_sig_candidate_transaciton
                            # (re)sign this candidate transaction
                            signing_time = miner.sign_candidate_transaction(transaction_to_sign)
                            new_begin_mining_time = arrival_time + validation_time + signing_time
                    else:
                        print(
                            f"A verification process is skipped for the transaction from miner {unconfirmmed_transaction['validation_done_by']} by miner {miner.return_idx()} due to miner offline.")
                        new_begin_mining_time = arrival_time
                    begin_mining_time = new_begin_mining_time if new_begin_mining_time > begin_mining_time else begin_mining_time
                transactions_to_record_in_block = {}
                transactions_to_record_in_block['valid_miner_sig_transacitons'] = valid_miner_sig_candidate_transacitons
                transactions_to_record_in_block['invalid_miner_sig_transacitons'] = invalid_miner_sig_candidate_transacitons
                # 将事务放入候选块并开始挖掘
                # block index starts from all-MiniLM-L6-v2
                start_time_point = time.time()
                candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length() + 1,
                                        transactions=transactions_to_record_in_block,
                                        miner_rsa_pub_key=miner.return_rsa_pub_key())
                # mine the block
                miner_computation_power = miner.return_computation_power()
                if not miner_computation_power:
                    block_generation_time_spent = float('inf')
                    miner.set_block_generation_time_point(float('inf'))
                    print(f"{miner.return_idx()} - miner mines a block in INFINITE time...")
                    continue
                recorded_transactions = candidate_block.return_transactions()
                print(len(recorded_transactions))
                if recorded_transactions['valid_miner_sig_transacitons'] or recorded_transactions['invalid_miner_sig_transacitons']:
                    print(f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} mining the block...")
                    # return the last block and add previous hash
                    last_block = miner.return_blockchain_object().return_last_block()
                    if last_block is None:
                        # will mine the genesis block
                        candidate_block.set_previous_block_hash(None)
                    else:
                        candidate_block.set_previous_block_hash(last_block.compute_hash(hash_entire_block=True))
                    # mine the candidate block by PoW, inside which the block_hash is also set
                    mined_block = miner.mine_block(candidate_block, rewards)
                else:
                    print(recorded_transactions)
                    print("No transaction to mine for this block.")
                    continue
                # unfortunately may go offline while propagating its block
                if miner.online_switcher():
                    # sign the block
                    miner.sign_block(mined_block)
                    miner.set_mined_block(mined_block)
                    # record mining time
                    block_generation_time_spent = (time.time() - start_time_point) / miner_computation_power
                    miner.set_block_generation_time_point(begin_mining_time + block_generation_time_spent)
                    print(f"{miner.return_idx()} - miner mines a block in {block_generation_time_spent} seconds.")
                    # immediately propagate the block
                    miner.propagated_the_block(miner.return_block_generation_time_point(), mined_block)
                else:
                    print(
                        f"Unfortunately, {miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} goes offline after, if successful, mining a block. This if-successful-mined block is not propagated.")
            else:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} did not receive any transaction from miner or miner in this round.")

        print(''' Step 5 - 矿工决定是否添加一个传播块或自己开采的块作为合法块，并请求其关联设备下载该块''')
        forking_happened = False
        # comm_round_block_gen_time regarded as the time point when the winning miner mines its block, calculated from the beginning of the round. If there is forking in PoW or rewards info out of sync in PoS, this time is the avg time point of all the appended time by any device

        comm_round_block_gen_time = []
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            unordered_propagated_block_processing_queue = miner.return_unordered_propagated_block_processing_queue()
            # add self mined block to the processing queue and sort by time
            this_miner_mined_block = miner.return_mined_block()
            if this_miner_mined_block:
                unordered_propagated_block_processing_queue[
                    miner.return_block_generation_time_point()] = this_miner_mined_block
            ordered_all_blocks_processing_queue = sorted(unordered_propagated_block_processing_queue.items())
            if ordered_all_blocks_processing_queue:
                if mining_consensus == 'PoW':
                    print("\nselect winning block based on PoW")
                    # abort mining if propagated block is received
                    print(
                        f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} is deciding if a valid propagated block arrived before it successfully mines its own block...")
                    for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
                        verified_block, verification_time = miner.verify_block(block_to_verify,
                                                                               block_to_verify.return_mined_by())
                        if verified_block:
                            block_mined_by = verified_block.return_mined_by()
                            if block_mined_by == miner.return_idx():
                                print(f"Miner {miner.return_idx()} is adding its own mined block.")
                            else:
                                print(
                                    f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()}.")
                            if miner.online_switcher():
                                miner.add_block(verified_block)
                            else:
                                print(
                                    f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain.")
                            if miner.return_the_added_block():
                                # requesting devices in its associations to download this block
                                miner.request_to_download(verified_block, block_arrival_time + verification_time)
                                break
                else:
                    # PoS
                    candidate_PoS_blocks = {}
                    print("根据PoS选择获胜区块")
                    # filter the ordered_all_blocks_processing_queue to contain only the blocks within time limit
                    for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
                        if block_arrival_time < args['miner_pos_propagated_block_wait_time']:
                            candidate_PoS_blocks[devices_in_network.devices_set[
                                block_to_verify.return_mined_by()].return_stake()] = block_to_verify
                    high_to_low_stake_ordered_blocks = sorted(candidate_PoS_blocks.items(), reverse=True)
                    # for PoS, requests every device in the network to add a valid block that has the most miner stake in the PoS candidate blocks list, which can be verified through chain
                    for (stake, PoS_candidate_block) in high_to_low_stake_ordered_blocks:
                        verified_block, verification_time = miner.verify_block(PoS_candidate_block,
                                                                               PoS_candidate_block.return_mined_by())
                        if verified_block:
                            block_mined_by = verified_block.return_mined_by()
                            if block_mined_by == miner.return_idx():
                                print(f"Miner {miner.return_idx()} with stake {stake} 正在添加自己开采的区块.")
                            else:
                                print(
                                    f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()} with stake {stake}.")
                            if miner.online_switcher():
                                miner.add_block(verified_block)
                            else:
                                print(
                                    f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain.")
                            if miner.return_the_added_block():
                                # requesting devices in its associations to download this block
                                miner.request_to_download(verified_block, block_arrival_time + verification_time)
                                break
                miner.add_to_round_end_time(block_arrival_time + verification_time)
            else:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} does not receive a propagated block and has not mined its own block yet.")
        # CHECK FOR FORKING
        added_blocks_miner_set = set()
        for device in devices_list:
            the_added_block = device.return_the_added_block()
            if the_added_block:
                print(
                    f"{device.return_role()} {device.return_idx()} has added a block mined by {the_added_block.return_mined_by()}")
                added_blocks_miner_set.add(the_added_block.return_mined_by())
                block_generation_time_point = devices_in_network.devices_set[
                    the_added_block.return_mined_by()].return_block_generation_time_point()
                # commented, as we just want to plot the legitimate block gen time, and the wait time is to avoid forking. Also the logic is wrong. Should track the time to the slowest worker after its global model update
                # if mining_consensus == 'PoS':
                # 	if args['miner_pos_propagated_block_wait_time'] != float("inf"):
                # 		block_generation_time_point += args['miner_pos_propagated_block_wait_time']
                comm_round_block_gen_time.append(block_generation_time_point)
        if len(added_blocks_miner_set) > 1:
            print("WARNING: a forking event just happened!")
            forking_happened = True
            with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a') as file:
                file.write(f"Forking in round {comm_round}\n")
        else:
            print("No forking event happened.")

        print(
            ''' 步骤6最后一步-处理添加的块 - all-MiniLM-L6-v2.收集可用的更新参数\n 2.恶意节点识别\n 3.得到奖励\n 4.进行本地更新\n 如果在此轮中没有生成有效的代码块，则跳过该代码块''')
        all_devices_round_ends_time = []
        for device in devices_list:
            if device.return_the_added_block() and device.online_switcher():
                # collect usable updated params, malicious nodes identification, get rewards and do local udpates
                processing_time = device.process_block(device.return_the_added_block(), log_files_folder_path, conn,
                                                       conn_cursor)
                device.other_tasks_at_the_end_of_comm_round(comm_round, log_files_folder_path)
                device.add_to_round_end_time(processing_time)
                all_devices_round_ends_time.append(device.return_round_end_time())

        print(''' 按设备划分的日志记录精度 ''')
        for device in devices_list:
            device.accuracy_this_round = device.validate_model_weights()
            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                file.write(
                    f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.accuracy_this_round}\n")

        # logging time, mining_consensus and forking
        # get the slowest device end time
        comm_round_spent_time = time.time() - comm_round_start_time
        with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
            # corner case when all miners in this round are malicious devices so their blocks are rejected
            try:
                comm_round_block_gen_time = max(comm_round_block_gen_time)
                file.write(f"comm_round_block_gen_time: {comm_round_block_gen_time}\n")
            except:
                no_block_msg = "这一轮没有生成有效的区块."
                print(no_block_msg)
                file.write(f"comm_round_block_gen_time: {no_block_msg}\n")
                with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a') as file2:
                    # TODO this may be caused by "no transaction to mine" for the miner. Forgot to check for block miner's maliciousness in request_to_downlaod()
                    file2.write(f"No valid block in round {comm_round}\n")
            try:
                slowest_round_ends_time = max(all_devices_round_ends_time)
                file.write(f"slowest_device_round_ends_time: {slowest_round_ends_time}\n")
            except:
                # corner case when all transactions are rejected by miners
                file.write("slowest_device_round_ends_time: No valid block has been generated this round.\n")
                with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'r+') as file2:
                    no_valid_block_msg = f"No valid block in round {comm_round}\n"
                    if file2.readlines()[-1] != no_valid_block_msg:
                        file2.write(no_valid_block_msg)
            file.write(f"mining_consensus: {mining_consensus} {args['pow_difficulty']}\n")
            file.write(f"forking_happened: {forking_happened}\n")
            file.write(f"comm_round_spent_time_on_this_machine: {comm_round_spent_time}\n")
        conn.commit()

        # if no forking, log the block miner
        if not forking_happened:
            legitimate_block = None
            for device in devices_list:
                legitimate_block = device.return_the_added_block()
                if legitimate_block is not None:
                    # skip the device who's been identified malicious and cannot get a block from miners
                    break
            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                if legitimate_block is None:
                    file.write("block_mined_by: no valid block generated this round\n")
                else:
                    block_mined_by = legitimate_block.return_mined_by()
                    is_malicious_node = "M" if devices_in_network.devices_set[
                        block_mined_by].return_is_malicious() else "B"
                    file.write(f"block_mined_by: {block_mined_by} {is_malicious_node}\n")
        else:
            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                file.write(f"block_mined_by: Forking happened\n")

        print(''' Logging Stake by Devices ''')
        for device in devices_list:
            device.accuracy_this_round = device.validate_model_weights()
            with open(f"{log_files_folder_path_comm_round}/stake_comm_{comm_round}.txt", "a") as file:
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                file.write(
                    f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.return_stake()}\n")

        # a temporary workaround to free GPU mem by delete txs stored in the blocks. Not good when need to resync chain
        if args['destroy_tx_in_block']:
            for device in devices_list:
                last_block = device.return_blockchain_object().return_last_block()
                if last_block:
                    last_block.free_tx()

        # save network_snapshot if reaches save frequency
        if args['save_network_snapshots'] and (comm_round == 1 or comm_round % args['save_freq'] == 0):
            if args['save_most_recent']:
                paths = sorted(Path(network_snapshot_save_path).iterdir(), key=os.path.getmtime)
                if len(paths) > args['save_most_recent']:
                    for _ in range(len(paths) - args['save_most_recent']):
                        # make it 0 byte as os.remove() moves file to the bin but may still take space
                        # https://stackoverflow.com/questions/53028607/how-to-remove-the-file-from-trash-in-drive-in-colab
                        open(paths[_], 'w').close()
                        os.remove(paths[_])
            snapshot_file_path = f"{network_snapshot_save_path}/snapshot_r_{comm_round}"
            print(f"Saving network snapshot to {snapshot_file_path}")
            pickle.dump(devices_in_network, open(snapshot_file_path, "wb"))

#奥利弗