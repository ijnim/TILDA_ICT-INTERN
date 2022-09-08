# $Id: ga.m,v 1.2 2006/09/01 17:06:46 yschoe Exp yschoe $
#
# Author: Yoonsuck Choe choe(a)tamu.edu
#	http://faculty.cs.tamu.edu
#
# License: GNU Public License -- see http://www.gnu.org
#
# GA test code, for supervised learning of feedforward neural networks with
# one hidden layer.
#
# function [gene_pool,gen_err,min_net] = ga(input,target,params)
#
#	input is a matrix, where each row is one instance vector.
#	target is a matrix, where each row is one target vector.
#
#	params.n_hidden is the number of hidden units (integer, e.g., 10)
#	params.n_individuals is the population size (integer, e.g., 500)
#	params.retention_rate is the retention rate of top performers (0.0-1.0)
#	params.mutation_rate is the mutation rate (0.0-1.0)
#	params.max_generation is the max number of generations before halting
#	params.mutation_magnitude is the scaling factor for mutation (0.0-1.0)
#	params.error_thresh is the error threshold for halting (e.g., 0.001)
#
#	returns:
#		gene_pool: array of all individuals (network weights)
#		gen_err: min error in each generation
#		min_net: index of best individual (in gene_pool).
#
#			gene_pool(min_net).input_to_hidden
#			gene_pool(min_net).hidden_to_output
#
#			gives the weight matrices of the best individual.
#
# Example1: learn XOR
#
# 파라미터 설정
p.n_hidden=10; # hidden layer의 뉴런 수
p.n_hidden=10; # hidden layer의 뉴런 수
p.n_individuals=100; # 하나의 generation 안의 population 수(?)
p.retention_rate=0.2; # retention 비율
p.mutation_rate=0.6; # mutation 비율
p.max_generation=50; # 최대 generation 수 (50을 넘어가면 중단)
p.mutation_magnitude=0.5; # ??? mutation을 위한 스케일링 요소가 무슨 뜻인지?
                          # -> 얼만큼 mutation 을 해줄건지 그 광도
p.error_thresh=0.001; # 0.001보다 작은 차이가 나면 중단
params = p;

#input, output 값을 받아 함수 실행
[gp,err,midx] = ga([-1 -1; -1 1; 1 -1; 1 1], [-1; 1; 1; -1], p);
#
#
# ga 함수 정의
function [gene_pool,gen_err,min_net] = ga(rawinput,target,params)
n_individuals = params.n_individuals;
#----------------------------------------
# neural network configuration
#----------------------------------------
n_output= columns(target);
n_hidden= params.n_hidden+1; 		# "+1" for bias unit
n_input = columns(rawinput)+1;		# "+1" for bias unit
#----------------------------------------
### other constants
#----------------------------------------
input_count = rows(rawinput);
#----------------------------------------
# setup bias
#----------------------------------------
input = [rawinput, -ones(input_count,1)]; # 4개의 input + -1 (bias)
#----------------------------------------
# initialize population (the connection weights)
#----------------------------------------
for i=1:n_individuals
	# initialize weights to small random values.
  # random하게 chromosome 생성
  # 2차원 벡터로 initialize -> normalize
	gene_pool(i).input_to_hidden = rand(n_input,n_hidden);
	gene_pool(i).input_to_hidden -= mean(vec(gene_pool(i).input_to_hidden));
	gene_pool(i).hidden_to_output = rand(n_hidden,n_output);
	gene_pool(i).hidden_to_output -= mean(vec(gene_pool(i).hidden_to_output));
end
#----------------------------------------
# Main loop
#----------------------------------------
# 1st stopping criteria
for g=1:params.max_generation
#--------------------
# 1. calculate fitness
#--------------------
break_flag=0;
for i=1:n_individuals

	err(i) = 0;
	for j=1:input_count
		# simple network activation
    # 생성된 individual로 네트워크 생성
		hidden = tanh(input(j,:)*gene_pool(i).input_to_hidden); #phenotype이 network이기 때문에 activation function 생성
		hidden(n_hidden)=-1; # bias unit
		output = tanh(hidden*gene_pool(i).hidden_to_output);
		err(i) += sum((target(j,:)-output).^2)/n_output;
	end
	# mean of sum squared error
	err(i)=err(i)/input_count;
	# find best individual
  # find best individual이 아니라,
  # 2nd stopping criteria를 검증해보기 위해 err가 가장 작은 값을 뽑아 확인한 것 같음
	[gen_err(g),min_net] = min(err);

	# halt if error is below threshold
  # 2nd stopping criteria
	if gen_err(g)<params.error_thresh
		break_flag=1;
	end
end
if break_flag==1
   break;
end
#
# plot fitness of the whole population
#
eval(sprintf("ylim([%f:2])",params.error_thresh));
plot(gen_err);
#--------------------
# 2. select top performers
#--------------------
# error가 작은 값을 기준으로 정렬하여 best individual 선정
[score,network_id]=sort(err);
n_retain = floor(n_individuals * params.retention_rate); #얼마나 선택할 것인가
for i=1:n_retain
	gene_pool(i).input_to_hidden = gene_pool(network_id(i)).input_to_hidden;
	gene_pool(i).hidden_to_output = gene_pool(network_id(i)).hidden_to_output;
end
#--------------------
# 3. cross over
#--------------------
for i=n_retain+1:n_individuals
	#----------
	# Select parents randomly
	#----------
	dad_id = 0;
	mom_id = network_id(ceil(rand*(n_retain)));
	while (mom_id != dad_id)
		dad_id = network_id(ceil(rand*(n_retain)));
	end
	#----------
	# Hidden
	#----------
  # one-point crosover
	cross_over = ceil(rand*n_input);
	offspring = [gene_pool(mom_id).input_to_hidden(1:cross_over,:);
	             gene_pool(dad_id).input_to_hidden(cross_over+1:n_input,:)];
	gene_pool(i).input_to_hidden = offspring;
	#----------
	# Output
	#----------
	cross_over = ceil(rand*n_hidden);
	offspring = [gene_pool(mom_id).hidden_to_output(1:cross_over,:);
	             gene_pool(dad_id).hidden_to_output(cross_over+1:n_hidden,:)];
	gene_pool(i).hidden_to_output = offspring;
end
#--------------------
# 4. mutate
#--------------------
for i=1:n_individuals
	#----------
 	# Set mutation scale to be different between retained ones and their
	# offsprings.
	#----------
	if (i<=n_retain)
		scale=0.2; # retained ones have lower mutation rate
               # ??? 선택된 애들한테 더 낮은 비율을 주는 이유는..?
               # 선택된 애들이 우수한 애들이니까 변이를 적게 주는 것으로 추정
	else
		scale=1.0;
	end
	#----------
	# Hidden
	#----------
  # mask : 특정 weight만 선택하여 랜덤하게 조금 변형해주기
	mask = rand(n_input,n_hidden)>(1-params.mutation_rate);
	rmask = mask.*(0.5-rand(n_input,n_hidden))*params.mutation_magnitude*scale;
	smask = sum(vec(mask));
	if (smask != 0)
	  m = sum(vec(rmask))/sum(vec(mask));
	  gene_pool(i).input_to_hidden = gene_pool(i).input_to_hidden+(rmask);
	end
	#----------
	# Output
	#----------
	mask = rand(n_hidden,n_output)>(1-params.mutation_rate);
	rmask = mask.*(0.5-rand(n_hidden,n_output))*params.mutation_magnitude*scale;
	smask = sum(vec(mask));
	if (smask != 0)
	  m = sum(vec(rmask))/sum(vec(mask));
	  gene_pool(i).hidden_to_output = gene_pool(i).hidden_to_output+(rmask);
	end
end
end # end of main loop
end # end of function
