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
#pow_resync_chain
#update_model_after_chain_resync
# TODO miner sometimes receives worker transactions directly for unknown reason - discard tx if it's not the correct type
# TODO a chain is invalid if a malicious block is identified after this miner is identified as malicious
# TODO Do not associate with blacklisted node. This may be done already.
# TODO KickR continuousness should skip the rounds when nodes are not selected as workers
# TODO update forking log after loading network snapshots
# TODO in reuqest_to_download, forgot to check for maliciousness of the block miner
# future work
# TODO - non-even dataset distribution
import os
os.environ["KMP_DUPLICATE_LIB_O"]="TRUE"

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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Block_FedAvg_Simulation")

# debug attributes
# debug 属性
parser.add_argument('-g', '--gpu', type=str, default='all-MiniLM-L6-v2', help='gpu id to use(e.g. 0,all-MiniLM-L6-v2,2,3)')
parser.add_argument('-v', '--verbose', type=int, default=1, help='打印详细调试日志')
parser.add_argument('-sn', '--save_network_snapshots', type=int, default=0, help='如果设置为1，只保存network_snapshots; 将在快照文件夹中创建一个带有日期的文件夹')
parser.add_argument('-dtx', '--destroy_tx_in_block', type=int, default=0, help='目前，存储在块中的事务占用GPU内存，并且还没有找到将它们移动到CPU内存或硬盘的方法, 所以打开它来节省GPU内存，以便PoS运行100+轮。如果需要执行链重同步，则不太好。')
parser.add_argument('-rp', '--resume_path', type=str, default=None, help='从保存network_snapshots的路径恢复；只提供日期')
parser.add_argument('-sf', '--save_freq', type=int, default=5, help='network_snapshot的保存频率')
parser.add_argument('-sm', '--save_most_recent', type=int, default=2, help='为了节省空间，只保留最近指定数量的快照；0表示保留全部')

# FL attributes
# 联邦学习属性
parser.add_argument('-B', '--batchsize', type=int, default=10, help='本地训练尺寸')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='训练的模型')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="学习率, use value from origin paper as default")
parser.add_argument('-op', '--optimizer', type=str, default="SGD", help='要使用的优化器，默认情况下实现随机梯度下降')
parser.add_argument('-iid', '--IID', type=int, default=0, help='将数据分配给设备的方式')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=100, help='最大通信轮数，如果收敛可能提前终止')
parser.add_argument('-nd', '--num_devices', type=int, default=20, help='模拟网络中设备的数量')
parser.add_argument('-st', '--shard_test_data', type=int, default=0, help='当测试数据集没有分片时，很容易看到全局模型在设备之间是一致的')
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help="网络中恶意节点的数量。恶意节点的数据集将引入高斯噪声")
parser.add_argument('-nv', '--noise_variance', type=int, default=1, help="注入高斯噪声的噪声方差水平")
parser.add_argument('-le', '--default_local_epochs', type=int, default=5, help='本地训练. 如果未指定-mt，则对每个worker按相同的epoch数训练本地模型')

# blockchain system consensus attributes
# 区块链系统共识属性

parser.add_argument('-ur', '--unit_reward', type=int, default=1, help='一系列的奖励unit reward for providing data, verification of signature, validation and so forth')
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=6, help="如果在此轮数中被识别为恶意设备，则工作器或验证器设备将被踢出设备的对等列表（放入黑名单）")
parser.add_argument('-lo', '--lazy_worker_knock_out_rounds', type=int, default=10, help="如果一个工作设备没有为这个轮数提供更新，那么它就会被踢出设备的对等体列表（放入黑名单），因为它太慢或懒得进行更新，只接受模型更新。（不要在意懒惰的验证者或矿工，因为他们不会得到奖励）")
parser.add_argument('-pow', '--pow_difficulty', type=int, default=0, help="如果设置为0，则表示矿工正在使用PoS")

# blockchain FL validator/miner restriction tuning parameters
# 区块链FL验证器/矿工限制调优参数
parser.add_argument('-mt', '--miner_acception_wait_time', type=float, default=0.0, help="矿工接受交易的默认时间窗口，以秒为单位。0表示没有时间限制，每个设备每轮只执行相同数量（-le）的epoch，就像fedag论文中一样")
parser.add_argument('-ml', '--miner_accepted_transactions_size_limit', type=float, default=0.0, help="在此限制之后，矿工将不再接受任何交易。0表示没有大小限制。必须指定this或-mt，或者两者都指定。这个参数决定了最终的block_size")
parser.add_argument('-mp', '--miner_pos_propagated_block_wait_time', type=float, default=float("inf"), help="此等待时间从通信轮开始计算，用于模拟PoS中的分叉事件")
parser.add_argument('-vh', '--validator_threshold', type=float, default=1.0, help="一个准确度的阈值来判断恶意工作者")
parser.add_argument('-md', '--malicious_updates_discount', type=float, default=0.0, help="do not entirely drop the voted negative worker transaction because that risks the same worker dropping the entire transactions and repeat its accuracy again and again and will be kicked out. Apply a discount factor instead to the false negative worker's updates are by some rate applied so it won't repeat")
parser.add_argument('-mv', '--malicious_validator_on', type=int, default=0, help="让恶意验证器翻转投票结果")


# distributed system attributes
# 分布式系统属性
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='设备在线的几率')
parser.add_argument('-els', '--even_link_speed_strength', type=int, default=1, help="该变量用于模拟传输延迟。缺省值1表示为每个设备分配相同的链路速度强度-dts字节/秒。如果设置为0，则链路速度强度在0到1之间随机启动，这意味着设备在实验期间将传输-els*-dts字节/秒，一次交易约为35k字节。")
parser.add_argument('-dts', '--base_data_transmission_speed', type=float, default=70000.0, help="当-els == 1时，每秒可传输的数据量。设置这个变量来决定传输速度（带宽），这进一步决定了传输延迟——在实验中，一个事务大约是35k字节。")
parser.add_argument('-ecp', '--even_computation_power', type=int, default=1, help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value all-MiniLM-L6-v2 means evenly assign computation power to all-MiniLM-L6-v2. If set to 0, power is randomly initiated as an int between 0 and 4, both included.")

# simulation attributes
# 仿真属性
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*', help="hard assign number of roles in the network, order by worker, validator and miner. e.g. 12,5,3 assign 12 workers, 5 validators and 3 miners. \"*,*,*\" means completely random role-assigning in each communication round ")
parser.add_argument('-aio', '--all_in_one', type=int, default=1, help='让所有节点在注册时都知道网络中的彼此')
parser.add_argument('-cs', '--check_signature', type=int, default=1, help='如果设置为0，则假定所有签名都经过验证，以节省执行时间')

# parser.add_argument('-la', '--least_assign', type=str, default='*,*,*', help='the assigned number of roles are at least guaranteed in the network')

if __name__=="__main__":

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
		latest_network_snapshot_file_name = sorted([f for f in os.listdir(network_snapshot_save_path) if not f.startswith('.')], key = lambda fn: int(fn.split('_')[-1]) , reverse=True)[0]
		print(f"Loading network snapshot from {args['resume_path']}/{latest_network_snapshot_file_name}")
		print("注意：加载的dev env必须与当前的dev env相同，即cpu、gpu或gpu并行")
		latest_round_num = int(latest_network_snapshot_file_name.split('_')[-1])
		devices_in_network = pickle.load(open(f"{network_snapshot_save_path}/{latest_network_snapshot_file_name}", "rb"))
		devices_list = list(devices_in_network.devices_set.values())
		log_files_folder_path = f"logs/{args['resume_path']}"
		# for colab
		# log_files_folder_path = f"/content/drive/MyDrive/BFA/logs/{args['resume_path']}"
		# original arguments file
		args_used_file = f"{log_files_folder_path}/args_used.txt"
		file = open(args_used_file,"r") 
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
			validators_needed = int(roles_requirement[1])
		except:
			validators_needed = 1
		try:
			miners_needed = int(roles_requirement[2])
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
			validators_needed = int(roles_requirement[1])
		except:
			validators_needed = 1
		try:
			miners_needed = int(roles_requirement[2])
		except:
			miners_needed = 1

		# 5. check arguments eligibility（检查参数合格性）

		num_devices = args['num_devices']
		num_malicious = args['num_malicious']
		
		if num_devices < workers_needed + miners_needed + validators_needed:
			sys.exit("ERROR: 分配给设备的角色超过了网络中允许的最大设备数量.")

		if num_devices < 3:
			sys.exit("ERROR: 网络中设备不足.\n The system needs at least one miner, one worker and/or one validator to start the operation.\nSystem aborted.")

		
		if num_malicious:
			if num_malicious > num_devices:
				sys.exit("ERROR: 恶意节点数量不能超过该网络中设置的设备总数")
			else:
				print(f"恶意节点vs设备总数设置为 {num_malicious}/{num_devices} = {(num_malicious/num_devices)*100:.2f}%")

		# 6. 根据输入模型名创建神经网络
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
		devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=args['IID'], batch_size = args['batchsize'], learning_rate =  args['learning_rate'], loss_func = loss_func, opti = args['optimizer'], num_devices=num_devices, network_stability=args['network_stability'], net=net, dev=dev, knock_out_rounds=args['knock_out_rounds'], lazy_worker_knock_out_rounds=args['lazy_worker_knock_out_rounds'], shard_test_data=args['shard_test_data'], miner_acception_wait_time=args['miner_acception_wait_time'], miner_accepted_transactions_size_limit=args['miner_accepted_transactions_size_limit'], validator_threshold=args['validator_threshold'], pow_difficulty=args['pow_difficulty'], even_link_speed_strength=args['even_link_speed_strength'], base_data_transmission_speed=args['base_data_transmission_speed'], even_computation_power=args['even_computation_power'], malicious_updates_discount=args['malicious_updates_discount'], num_malicious=num_malicious, noise_variance=args['noise_variance'], check_signature=args['check_signature'], not_resync_chain=args['destroy_tx_in_block'])
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
		# open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'w').close()
		# open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'w').close()
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
	for comm_round in range(latest_round_num + 1, args['max_num_comm']+1):
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
		validators_to_assign = validators_needed
		workers_this_round = []
		miners_this_round = []
		validators_this_round = []
		random.shuffle(devices_list)
		# 打乱网络设备顺序后对各个设备分配工作
		for device in devices_list:
			if workers_to_assign:
				device.assign_worker_role()
				workers_to_assign -= 1
			elif miners_to_assign:
				device.assign_miner_role()
				miners_to_assign -= 1
			elif validators_to_assign:
				device.assign_validator_role()
				validators_to_assign -= 1
			else:
				device.assign_role()
				# 随机分配
			if device.return_role() == 'worker':
				workers_this_round.append(device)
			elif device.return_role() == 'miner':
				miners_this_round.append(device)
			else:
				validators_this_round.append(device)
			# determine if online at the beginning (essential for step all-MiniLM-L6-v2 when worker needs to associate with an online device)
			# 确定在开始时是否在线（当worker需要与在线设备关联时，对于第1步至关重要）
			# 用随机数的方式随机生成，来表示设备的在线状态，这里对每一个设备进行设置
			device.online_switcher()
		# 创建日志文件夹，随机分配三种角色给所有人

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
			print(f"\n在本轮训练中 {len(workers_this_round)}名工人, {len(miners_this_round)} 名矿工 and {len(validators_this_round)} 名验证者 .")
			print("\n这一轮的工人是")
			for worker in workers_this_round:
				print(f"d_{worker.return_idx().split('_')[-1]} online - {worker.is_online()} with chain len {worker.return_blockchain_object().return_chain_length()}")
			print("\n这一轮的矿工是")
			for miner in miners_this_round:
				print(f"d_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_blockchain_object().return_chain_length()}")
			print("\n这一轮的验证者是")
			for validator in validators_this_round:
				print(f"d_{validator.return_idx().split('_')[-1]} online - {validator.is_online()} with chain len {validator.return_blockchain_object().return_chain_length()}")
			print()

			# show peers with round number
			# 未懂代码
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
		for validator in validators_this_round:
			validator.validator_reset_vars_for_new_round()
		# 清除状态开始全新一轮

		# DOESN'T MATTER ANY MORE AFTER TRACKING TIME, but let's keep it - orginal purpose: shuffle the list(for worker, this will affect the order of dataset portions to be trained)
		# 跟踪时间结束后不再重要，但让我们保留它 - 最初的目的：洗牌列表（对于worker，这将影响要训练的数据集部分的顺序）
		random.shuffle(workers_this_round)
		random.shuffle(miners_this_round)
		random.shuffle(validators_this_round)
		
		''' 工人、验证者和矿工轮流完成工作 '''

		print(''' 步骤1 -工人分配相关的矿工和验证器(并进行本地更新，但它是在步骤2的代码块中实现的) \n''')
		for worker_iter in range(len(workers_this_round)):
			worker = workers_this_round[worker_iter]
			# resync chain(block could be dropped due to fork from last round)
			# 重新同步链（由于上一轮的分叉，区块可能会被丢弃）
			if worker.resync_chain(mining_consensus):
				worker.update_model_after_chain_resync(log_files_folder_path_comm_round, conn, conn_cursor)
			# worker (should) perform local update and associate
			# Worker（应该）执行本地更新和关联
			print(f"{worker.return_idx()} - worker {worker_iter+1}/{len(workers_this_round)} 工人节点将尝试与验证者和矿工建立关联，以便进行后续的交易和更新传递...")
			# worker associates with a miner to accept finally mined block
			# 工人与矿工合作接受最终开采的区块
			if worker.online_switcher():
				associated_miner = worker.associate_with_device("miner")
				if associated_miner:
					associated_miner.add_device_to_association(worker)
				else:
					print(f"没找到一个合适的矿工节点（miner）为了该 {worker.return_idx()}id的工作节点 peer list.")
			# worker associates with a validator to send worker transactions
			# Worker与验证器关联以发送Worker事务
			if worker.online_switcher():
				associated_validator = worker.associate_with_device("validator")
				if associated_validator:
					associated_validator.add_device_to_association(worker)
				else:
					print(f"没找到一个合适的验证者节点（validator）{worker.return_idx()}id的工作节点 peer list.")

		# 	为工人分配相关的矿工节点和验证者节点
		
		print(''' 步骤2 -验证器接受本地更新，并将其广播到各自对等列表中的其他验证器（在此步骤中调用工作器local_updates()）。\n''')
		for validator_iter in range(len(validators_this_round)):
			validator = validators_this_round[validator_iter]
			# resync chain
			# 同步链
			if validator.resync_chain(mining_consensus):
				validator.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
			# associate with a miner to send post validation transactions
			# 与矿工关联以发送验证后交易
			if validator.online_switcher():
				associated_miner = validator.associate_with_device("miner")
				if associated_miner:
					associated_miner.add_device_to_association(validator)
				else:
					print(f"没有找到一个合适的矿工节点（miner）为了该{validator.return_idx()}id验证者 peer list.")
			# validator accepts local updates from its workers association
			# 验证器接受来自其工作者关联的本地更新
			associated_workers = list(validator.return_associated_workers())
			if not associated_workers:
				print(f"在这一轮验证者沟通中，没有找到相关联的工作者 {validator.return_idx()} {validator_iter+1}/{len(validators_this_round)} .")
				continue
			validator_link_speed = validator.return_link_speed()
			print(f"如果在线的话，{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} 是否接受工人的更新与链接速度{validator_link_speed} 字节/秒")
			# records_dict used to record transmission delay for each epoch to determine the next epoch updates arrival time
			# Records_dict用于记录每个epoch的传输延迟，以确定下一个epoch更新的到达时间
			records_dict = dict.fromkeys(associated_workers, None)
			for worker, _ in records_dict.items():
				records_dict[worker] = {}
			# used for arrival time easy sorting for later validator broadcasting (and miners' acception order)
			# 用于到达时间的简单排序，以便稍后的验证器广播（和矿工的接受顺序）
			transaction_arrival_queue = {}
			# workers local_updates() called here as their updates transmission may be restrained by miners' acception time and/or size
			# 这里调用的# workers local_updates（）作为他们的更新传输可能受到矿工接受时间和/或大小的限制
			if args['miner_acception_wait_time']:
				print(f"矿工等待时间指定为{args['miner_acception_wait_time']} 秒。让每个worker执行local_updates直到时间限制")
				for worker_iter in range(len(associated_workers)):
					worker = associated_workers[worker_iter]
					if not worker.return_idx() in validator.return_black_list():
						# TODO here, also add print() for below miner's validators
						print(f'worker {worker_iter+1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates')	 
						total_time_tracker = 0
						update_iter = 1
						worker_link_speed = worker.return_link_speed()
						lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
						while total_time_tracker < validator.return_miner_acception_wait_time():
							# simulate the situation that worker may go offline during model updates transmission to the validator, based on per transaction
							# 基于每个事务，模拟worker在模型更新传输到验证器期间可能脱机的情况
							if worker.online_switcher():
								local_update_spent_time = worker.worker_local_update(rewards, log_files_folder_path_comm_round, comm_round)
								unverified_transaction = worker.return_local_updates_and_signature(comm_round)
								# size in bytes, usually around 35000 bytes per transaction
								# 大小（以字节为单位），每个事务通常在35000字节左右
								unverified_transactions_size = getsizeof(str(unverified_transaction))
								transmission_delay = unverified_transactions_size/lower_link_speed
								if local_update_spent_time + transmission_delay > validator.return_miner_acception_wait_time():
									# last transaction sent passes the acception time window
									# 最后发送的事务通过接受时间窗口
									break
								records_dict[worker][update_iter] = {}
								records_dict[worker][update_iter]['local_update_time'] = local_update_spent_time
								records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
								records_dict[worker][update_iter]['local_update_unverified_transaction'] = unverified_transaction
								records_dict[worker][update_iter]['local_update_unverified_transaction_size'] = unverified_transactions_size
								if update_iter == 1:
									total_time_tracker = local_update_spent_time + transmission_delay
								else:
									total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1]['transmission_delay'] + local_update_spent_time + transmission_delay
								records_dict[worker][update_iter]['arrival_time'] = total_time_tracker
								if validator.online_switcher():
									# accept this transaction only if the validator is online
									# 只有当验证器在线时才接受此事务
									print(f"validator {validator.return_idx()} has accepted this transaction.")
									transaction_arrival_queue[total_time_tracker] = unverified_transaction
								else:
									print(f"validator {validator.return_idx()} offline and unable to accept this transaction")
							else:
								# worker goes offline and skip updating for one transaction, wasted the time of one update and transmission
								# Worker脱机并跳过一个事务的更新，浪费了一次更新和传输的时间
								wasted_update_time, wasted_update_params = worker.waste_one_epoch_local_update_time(args['optimizer'])
								wasted_update_params_size = getsizeof(str(wasted_update_params))
								wasted_transmission_delay = wasted_update_params_size/lower_link_speed
								if wasted_update_time + wasted_transmission_delay > validator.return_miner_acception_wait_time():
									# wasted transaction "arrival" passes the acception time window
									# 浪费的事务“到达”通过了接受时间窗口
									break
								records_dict[worker][update_iter] = {}
								records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
								if update_iter == 1:
									total_time_tracker = wasted_update_time + wasted_transmission_delay
									print(f"worker goes offline and wasted {total_time_tracker} seconds for a transaction")
								else:
									total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1]['transmission_delay'] + wasted_update_time + wasted_transmission_delay
							update_iter += 1
			else:
				 # did not specify wait time. every associated worker perform specified number of local epochs
				 # 没有指定等	待时间。每个相关的工作执行指定数量的局部epoch
				for worker_iter in range(len(associated_workers)):
					worker = associated_workers[worker_iter]
					if not worker.return_idx() in validator.return_black_list():
						print(f'worker {worker_iter+1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates')	 
						if worker.online_switcher():
							local_update_spent_time = worker.worker_local_update(rewards, log_files_folder_path_comm_round, comm_round, local_epochs=args['default_local_epochs'])
							worker_link_speed = worker.return_link_speed()
							lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
							unverified_transaction = worker.return_local_updates_and_signature(comm_round)
							unverified_transactions_size = getsizeof(str(unverified_transaction))
							transmission_delay = unverified_transactions_size/lower_link_speed
							if validator.online_switcher():
								transaction_arrival_queue[local_update_spent_time + transmission_delay] = unverified_transaction
								print(f"validator {validator.return_idx()} has accepted this transaction.")
							else:
								print(f"validator {validator.return_idx()} offline and unable to accept this transaction")
						else:
							print(f"worker {worker.return_idx()} offline and unable do local updates")
					else:
						print(f"worker {worker.return_idx()} in validator {validator.return_idx()}'s black list. This worker's transactions won't be accpeted.")
			validator.set_unordered_arrival_time_accepted_worker_transactions(transaction_arrival_queue)
			# in case validator off line for accepting broadcasted transactions but can later back online to validate the transactions itself receives
			# 以防验证器脱机接受广播的事务，但稍后可以重新联机以验证自己接收的事务
			validator.set_transaction_for_final_validating_queue(sorted(transaction_arrival_queue.items()))
			
			# broadcast to other validators
			if transaction_arrival_queue:
				validator.validator_broadcast_worker_transactions()
			else:
				print("这个验证器没有接收到任何事务，可能是由于工作器和/或验证器脱机或在执行本地更新或传输更新时超时，或者所有工作器都在验证器的黑名单中。")


		print('''步骤2.5——通过广播的工作者事务，验证者决定最终的事务到达顺序 \n''')
		for validator_iter in range(len(validators_this_round)):
			validator = validators_this_round[validator_iter]
			accepted_broadcasted_validator_transactions = validator.return_accepted_broadcasted_worker_transactions()
			print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions...")
			accepted_broadcasted_transactions_arrival_queue = {}
			if accepted_broadcasted_validator_transactions:
				# calculate broadcasted transactions arrival time
				# 计算广播的事务到达时间
				self_validator_link_speed = validator.return_link_speed()
				for broadcasting_validator_record in accepted_broadcasted_validator_transactions:
					broadcasting_validator_link_speed = broadcasting_validator_record['source_validator_link_speed']
					lower_link_speed = self_validator_link_speed if self_validator_link_speed < broadcasting_validator_link_speed else broadcasting_validator_link_speed
					for arrival_time_at_broadcasting_validator, broadcasted_transaction in broadcasting_validator_record['broadcasted_transactions'].items():
						transmission_delay = getsizeof(str(broadcasted_transaction))/lower_link_speed
						accepted_broadcasted_transactions_arrival_queue[transmission_delay + arrival_time_at_broadcasting_validator] = broadcasted_transaction
			else:
				print(f"validator {validator.return_idx()} {validator_iter+1}/{len(validators_this_round)} did not receive any broadcasted worker transaction this round.")
			# mix the boardcasted transactions with the direct accepted transactions
			# 将board - cast事务与直接接受的事务混合
			final_transactions_arrival_queue = sorted({**validator.return_unordered_arrival_time_accepted_worker_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
			validator.set_transaction_for_final_validating_queue(final_transactions_arrival_queue)
			print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")


		print(''' Step 3 - 验证器按照事务到达时间的顺序进行自我验证和交叉验证（验证来自工作器的本地更新）.\n''')
		for validator_iter in range(len(validators_this_round)):
			validator = validators_this_round[validator_iter]
			final_transactions_arrival_queue = validator.return_final_transactions_validating_queue()
			if final_transactions_arrival_queue:
				# validator asynchronously does one epoch of update and validate on its own test set
				# 验证器异步地在它自己的测试集上执行一次更新和验证
				local_validation_time = validator.validator_update_model_by_one_epoch_and_validate_local_accuracy(args['optimizer'])
				print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} 验证器按照事务到达时间的顺序进行自我验证和交叉验证")
				for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
					if validator.online_switcher():
						# validation won't begin until validator locally done one epoch of update and validation(worker transactions will be queued)
						# 验证将不会开始，直到验证器本地完成更新和验证的一个epoch（工作事务将排队）。
						if arrival_time < local_validation_time:
							arrival_time = local_validation_time
						validation_time, post_validation_unconfirmmed_transaction = validator.validate_worker_transaction(unconfirmmed_transaction, rewards, log_files_folder_path, comm_round, args['malicious_validator_on'])
						if validation_time:
							validator.add_post_validation_transaction_to_queue((arrival_time + validation_time, validator.return_link_speed(), post_validation_unconfirmmed_transaction))
							print(f"A validation process has been done for the transaction from worker {post_validation_unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()}")
					else:
						print(f"A validation process is skipped for the transaction from worker {post_validation_unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()} due to validator offline.")
			else:
				print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} did not receive any transaction from worker or validator in this round.")

		print('''4.验证者将验证后交易发送给相关的矿工，矿工将这些交易广播给各自同行列表中的其他矿工\n''')
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			# resync chain
			if miner.resync_chain(mining_consensus):
				miner.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
			print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting validators' post-validation transactions...")
			associated_validators = list(miner.return_associated_validators())
			if not associated_validators:
				print(f"No validators are associated with miner {miner.return_idx()} for this communication round.")
				continue
			self_miner_link_speed = miner.return_link_speed()
			validator_transactions_arrival_queue = {}
			for validator_iter in range(len(associated_validators)):
				validator = associated_validators[validator_iter]
				print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(associated_validators)} of miner {miner.return_idx()} is sending signature verified transaction...")
				post_validation_transactions_by_validator = validator.return_post_validation_transactions_queue()
				post_validation_unconfirmmed_transaction_iter = 1
				for (validator_sending_time, source_validator_link_spped, post_validation_unconfirmmed_transaction) in post_validation_transactions_by_validator:
					if validator.online_switcher() and miner.online_switcher():
						lower_link_speed = self_miner_link_speed if self_miner_link_speed < source_validator_link_spped else source_validator_link_spped
						transmission_delay = getsizeof(str(post_validation_unconfirmmed_transaction))/lower_link_speed
						validator_transactions_arrival_queue[validator_sending_time + transmission_delay] = post_validation_unconfirmmed_transaction
						print(f"miner {miner.return_idx()} has accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()}")
					else:
						print(f"miner {miner.return_idx()} has not accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()} due to one of devices or both offline.")
					post_validation_unconfirmmed_transaction_iter += 1
			miner.set_unordered_arrival_time_accepted_validator_transactions(validator_transactions_arrival_queue)
			miner.miner_broadcast_validator_transactions()

		print(''' Step 4.5 - 通过广播的验证器交易，矿工决定最终的交易到达顺序\n ''')
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			accepted_broadcasted_validator_transactions = miner.return_accepted_broadcasted_validator_transactions()
			self_miner_link_speed = miner.return_link_speed()
			print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions...")
			accepted_broadcasted_transactions_arrival_queue = {}
			if accepted_broadcasted_validator_transactions:
				# calculate broadcasted transactions arrival time
				for broadcasting_miner_record in accepted_broadcasted_validator_transactions:
					broadcasting_miner_link_speed = broadcasting_miner_record['source_device_link_speed']
					lower_link_speed = self_miner_link_speed if self_miner_link_speed < broadcasting_miner_link_speed else broadcasting_miner_link_speed
					for arrival_time_at_broadcasting_miner, broadcasted_transaction in broadcasting_miner_record['broadcasted_transactions'].items():
						transmission_delay = getsizeof(str(broadcasted_transaction))/lower_link_speed
						accepted_broadcasted_transactions_arrival_queue[transmission_delay + arrival_time_at_broadcasting_miner] = broadcasted_transaction
			else:
				print(f"miner {miner.return_idx()} {miner_iter+1}/{len(miners_this_round)} did not receive any broadcasted validator transaction this round.")
			# mix the boardcasted transactions with the direct accepted transactions
			final_transactions_arrival_queue = sorted({**miner.return_unordered_arrival_time_accepted_validator_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
			miner.set_candidate_transactions_for_final_mining_queue(final_transactions_arrival_queue)
			print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")
		
		print(''' Step 5 - 矿工按照交易到达时间的顺序进行自我验证和交叉验证（验证验证者的签名），并根据限制大小将交易记录在候选区块中。同时挖掘并传播区块。\n''')
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			final_transactions_arrival_queue = miner.return_final_candidate_transactions_mining_queue()
			valid_validator_sig_candidate_transacitons = []
			invalid_validator_sig_candidate_transacitons = []
			begin_mining_time = 0
			new_begin_mining_time = begin_mining_time
			if final_transactions_arrival_queue:
				print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is verifying received validator transactions...")
				time_limit = miner.return_miner_acception_wait_time()
				size_limit = miner.return_miner_accepted_transactions_size_limit()
				for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
					if miner.online_switcher():
						if time_limit:
							if arrival_time > time_limit:
								break
						if size_limit:
							if getsizeof(str(valid_validator_sig_candidate_transacitons+invalid_validator_sig_candidate_transacitons)) > size_limit:
								break
						# verify validator signature of this transaction
						verification_time, is_validator_sig_valid = miner.verify_validator_transaction(unconfirmmed_transaction)
						if verification_time:
							if is_validator_sig_valid:
								validator_info_this_tx = {
								'validator': unconfirmmed_transaction['validation_done_by'],
								'validation_rewards': unconfirmmed_transaction['validation_rewards'],
								'validation_time': unconfirmmed_transaction['validation_time'],
								'validator_rsa_pub_key': unconfirmmed_transaction['validator_rsa_pub_key'],
								'validator_signature': unconfirmmed_transaction['validator_signature'],
								'update_direction': unconfirmmed_transaction['update_direction'],
								'miner_device_idx': miner.return_idx(),
								'miner_verification_time': verification_time,
								'miner_rewards_for_this_tx': rewards}
								# validator's transaction signature valid
								found_same_worker_transaction = False
								for valid_validator_sig_candidate_transaciton in valid_validator_sig_candidate_transacitons:
									if valid_validator_sig_candidate_transaciton['worker_signature'] == unconfirmmed_transaction['worker_signature']:
										found_same_worker_transaction = True
										break
								if not found_same_worker_transaction:
									valid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
									del valid_validator_sig_candidate_transaciton['validation_done_by']
									del valid_validator_sig_candidate_transaciton['validation_rewards']
									del valid_validator_sig_candidate_transaciton['update_direction']
									del valid_validator_sig_candidate_transaciton['validation_time']
									del valid_validator_sig_candidate_transaciton['validator_rsa_pub_key']
									del valid_validator_sig_candidate_transaciton['validator_signature']
									valid_validator_sig_candidate_transaciton['positive_direction_validators'] = []
									valid_validator_sig_candidate_transaciton['negative_direction_validators'] = []
									valid_validator_sig_candidate_transacitons.append(valid_validator_sig_candidate_transaciton)
								if unconfirmmed_transaction['update_direction']:
									valid_validator_sig_candidate_transaciton['positive_direction_validators'].append(validator_info_this_tx)
								else:
									valid_validator_sig_candidate_transaciton['negative_direction_validators'].append(validator_info_this_tx)
								transaction_to_sign = valid_validator_sig_candidate_transaciton
							else:
								# validator's transaction signature invalid
								invalid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
								invalid_validator_sig_candidate_transaciton['miner_verification_time'] = verification_time
								invalid_validator_sig_candidate_transaciton['miner_rewards_for_this_tx'] = rewards
								invalid_validator_sig_candidate_transacitons.append(invalid_validator_sig_candidate_transaciton)
								transaction_to_sign = invalid_validator_sig_candidate_transaciton
							# (re)sign this candidate transaction
							signing_time = miner.sign_candidate_transaction(transaction_to_sign)
							new_begin_mining_time = arrival_time + verification_time + signing_time
					else:
						print(f"A verification process is skipped for the transaction from validator {unconfirmmed_transaction['validation_done_by']} by miner {miner.return_idx()} due to miner offline.")
						new_begin_mining_time = arrival_time
					begin_mining_time = new_begin_mining_time if new_begin_mining_time > begin_mining_time else begin_mining_time
				transactions_to_record_in_block = {}
				transactions_to_record_in_block['valid_validator_sig_transacitons'] = valid_validator_sig_candidate_transacitons
				transactions_to_record_in_block['invalid_validator_sig_transacitons'] = invalid_validator_sig_candidate_transacitons
				# put transactions into candidate block and begin mining
				# block index starts from all-MiniLM-L6-v2
				start_time_point = time.time()
				candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length()+1, transactions=transactions_to_record_in_block, miner_rsa_pub_key=miner.return_rsa_pub_key())
				# mine the block
				miner_computation_power = miner.return_computation_power()
				if not miner_computation_power:
					block_generation_time_spent = float('inf')
					miner.set_block_generation_time_point(float('inf'))
					print(f"{miner.return_idx()} - miner mines a block in INFINITE time...")
					continue
				recorded_transactions = candidate_block.return_transactions()
				if recorded_transactions['valid_validator_sig_transacitons'] or recorded_transactions['invalid_validator_sig_transacitons']:
					print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} mining the block...")
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
					print("No transaction to mine for this block.")
					continue
				# unfortunately may go offline while propagating its block
				if miner.online_switcher():
					# sign the block
					miner.sign_block(mined_block)
					miner.set_mined_block(mined_block)
					# record mining time
					block_generation_time_spent = (time.time() - start_time_point)/miner_computation_power
					miner.set_block_generation_time_point(begin_mining_time + block_generation_time_spent)
					print(f"{miner.return_idx()} - miner mines a block in {block_generation_time_spent} seconds.")
					# immediately propagate the block
					miner.propagated_the_block(miner.return_block_generation_time_point(), mined_block)
				else:
					print(f"Unfortunately, {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} goes offline after, if successful, mining a block. This if-successful-mined block is not propagated.")
			else:
				print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} did not receive any transaction from validator or miner in this round.")

		print(''' Step 6 - 矿工决定是否添加一个传播块或自己开采的块作为合法块，并请求其关联设备下载该块''')
		forking_happened = False
		# comm_round_block_gen_time regarded as the time point when the winning miner mines its block, calculated from the beginning of the round. If there is forking in PoW or rewards info out of sync in PoS, this time is the avg time point of all the appended time by any device
		comm_round_block_gen_time = []
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			unordered_propagated_block_processing_queue = miner.return_unordered_propagated_block_processing_queue()
			# add self mined block to the processing queue and sort by time
			this_miner_mined_block = miner.return_mined_block()
			if this_miner_mined_block:
				unordered_propagated_block_processing_queue[miner.return_block_generation_time_point()] = this_miner_mined_block
			ordered_all_blocks_processing_queue = sorted(unordered_propagated_block_processing_queue.items())
			if ordered_all_blocks_processing_queue:
				if mining_consensus == 'PoW':
					print("\nselect winning block based on PoW")
					# abort mining if propagated block is received
					print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is deciding if a valid propagated block arrived before it successfully mines its own block...")
					for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
						verified_block, verification_time = miner.verify_block(block_to_verify, block_to_verify.return_mined_by())
						if verified_block:
							block_mined_by = verified_block.return_mined_by()
							if block_mined_by == miner.return_idx():
								print(f"Miner {miner.return_idx()} is adding its own mined block.")
							else:
								print(f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()}.")
							if miner.online_switcher():
								miner.add_block(verified_block)
							else:
								print(f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain.")
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
							candidate_PoS_blocks[devices_in_network.devices_set[block_to_verify.return_mined_by()].return_stake()] = block_to_verify
					high_to_low_stake_ordered_blocks = sorted(candidate_PoS_blocks.items(), reverse=True)
					# for PoS, requests every device in the network to add a valid block that has the most miner stake in the PoS candidate blocks list, which can be verified through chain
					for (stake, PoS_candidate_block) in high_to_low_stake_ordered_blocks:
						verified_block, verification_time = miner.verify_block(PoS_candidate_block, PoS_candidate_block.return_mined_by())
						if verified_block:
							block_mined_by = verified_block.return_mined_by()
							if block_mined_by == miner.return_idx():
								print(f"Miner {miner.return_idx()} with stake {stake} 正在添加自己开采的区块.")
							else:
								print(f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()} with stake {stake}.")
							if miner.online_switcher():
								miner.add_block(verified_block)
							else:
								print(f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain.")
							if miner.return_the_added_block():
								# requesting devices in its associations to download this block
								miner.request_to_download(verified_block, block_arrival_time + verification_time)
								break
				miner.add_to_round_end_time(block_arrival_time + verification_time)
			else:
				print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} does not receive a propagated block and has not mined its own block yet.")
		# CHECK FOR FORKING
		added_blocks_miner_set = set()
		for device in devices_list:
			the_added_block = device.return_the_added_block()
			if the_added_block:
				print(f"{device.return_role()} {device.return_idx()} has added a block mined by {the_added_block.return_mined_by()}")
				added_blocks_miner_set.add(the_added_block.return_mined_by())
				block_generation_time_point = devices_in_network.devices_set[the_added_block.return_mined_by()].return_block_generation_time_point()
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


		print(''' 步骤6最后一步-处理添加的块 - all-MiniLM-L6-v2.收集可用的更新参数\n 2.恶意节点识别\n 3.得到奖励\n 4.进行本地更新\n 如果在此轮中没有生成有效的代码块，则跳过该代码块''')
		all_devices_round_ends_time = []
		for device in devices_list:
			if device.return_the_added_block() and device.online_switcher():
				# collect usable updated params, malicious nodes identification, get rewards and do local udpates
				processing_time = device.process_block(device.return_the_added_block(), log_files_folder_path, conn, conn_cursor)
				device.other_tasks_at_the_end_of_comm_round(comm_round, log_files_folder_path)
				device.add_to_round_end_time(processing_time)
				all_devices_round_ends_time.append(device.return_round_end_time())

		print(''' 按设备划分的日志记录精度 ''')
		for device in devices_list:
			device.accuracy_this_round = device.validate_model_weights()
			with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
				is_malicious_node = "M" if device.return_is_malicious() else "B"
				file.write(f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.accuracy_this_round}\n")

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
					is_malicious_node = "M" if devices_in_network.devices_set[block_mined_by].return_is_malicious() else "B"
					file.write(f"block_mined_by: {block_mined_by} {is_malicious_node}\n")
		else:
			with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
				file.write(f"block_mined_by: Forking happened\n")

		print(''' Logging Stake by Devices ''')
		for device in devices_list:
			device.accuracy_this_round = device.validate_model_weights()
			with open(f"{log_files_folder_path_comm_round}/stake_comm_{comm_round}.txt", "a") as file:
				is_malicious_node = "M" if device.return_is_malicious() else "B"
				file.write(f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.return_stake()}\n")

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