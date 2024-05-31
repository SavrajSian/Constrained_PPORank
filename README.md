## PPORank
Deep reinforcement learning for personalized drug recommendation
## Dependecies
**Python**>=3.7.10; torch(1.18.1); yaml(0.1.7);
numpy(>=1.18.1); pandas(1.0.0); scipy(1.4.1); seaborn(0.8.1); tensorboard (1.15.5);
matplotlib(3.1.1); joblib(0.14.1); json5(0.8.5);jupyter(1.0.0); tqdm (4.42.0)
For drugs processing: deepchem (2.5.0); h5py(2.10.0); hdf5(1.10.4)

## PPORank usage
The implementation of PPORank is in "main.py":
(cpu version) :
The following is an exmaple for GDSC dataset when training with 16 actors, projection dimension of 100 without normalize y 
more details can be found in arguments.py

```
python main.py   --num_processes 16  --nlayers_deep 2 --Data GDSC_ALL --analysis FULL  --algo ppo --f 100  --normalize_y 
```
model logs dir : ./logs

model saved dir: ./Saved

model prediction saved dir: ./results

The clean data could be found in 

[Data Sharing](https://drive.google.com/drive/folders/1-YcEcRP6IObhT8ojes9L29Z54P-japjJ?usp=sharing)

## Scripts for runing the experiments
Download and preprocess the GDSC dataset:
```
python ./preprocess/load_dataset.py load_GDSC.txt
```
Split the data for training and testing, create folds for cross-validation and Pretrain the MF layers' weight
```
python prepare.py config.yaml

```
Runing the experiments on ppo (with config file "./configs/configS_base.yaml"):

```
python results.py > results_ppo.txt

```
PPO experiment with TCGA cohort

```
 python load_TCGA.py
 
 python results_TCGA.py ./TCGA/TCGA_BRCA.npz 
```




### mps changes:

* cuda to mps stuff
* changing doubles/float64s to float32s
* M0 to M0.item() for one particular bit
* in get_log_prob, selected drug ids needing squeezing for some stupid reason only on mps

running this on mps takes wayyyy too long, it takes 50s per batch, 48 batches so 40 mins per epoch. this is way too long given default num is 1000 (or 1400 idk), either way thats a month to train

commented added .float() in a bunch of places where it wasn't before


### adding constraints

in PPO_Agent.py, there is the update function. the surrogate loss fn is calculated and this is used to update NN params with loss.backward(). this is where we can add constraints.
would do something along the lines of:
```# Original loss function
original_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)

# Constraint penalty
constraint_penalty = compute_constraint_penalty(...)  # Define this based on your specific constraints

# New loss function with penalty for constraint violation
total_loss = original_loss + constraint_penalty_weight * constraint_penalty

# Compute gradients and update parameters
total_loss.backward()
```
#### considerations

Designing Constraint Penalties: 
The key challenge in adding constraints is designing penalty functions that effectively guide the model towards satisfying these constraints without overly compromising the original objectives of the policy and value function learning. The penalties should be differentiable to allow for gradient-based optimization.

Choosing Penalty Weights: The weights for the penalty terms (or the Lagrangian multipliers, if using that approach) are crucial hyperparameters. If the weights are too high, the model may focus too much on satisfying the constraints at the expense of performance on the primary task. If too low, the constraints may be ignored.

Algorithmic Adjustments: Depending on the nature of the constraints, might also need to adjust other parts of the PPO algorithm like the policy update rules or the way advantage is calculated. These adjustments could be necessary to ensure that the constraints are properly considered during the learning process.



#### novel approach stuff

As far as I can tell:
* No one using PPO for *ranking* problems has added constraints
  * There are a bunch of PPO constraint papers, PPO-Lagrangian seems our best bet. There is another paper building on this to make sure it doesnt get stuck in a local minima. 
  * There are also papers on formulating into unconstrained format - this is another possible route
  * For the penalty/constraint for side effects, the log-barrier function seems like a good bet
  * There are RL ranking with constraints papers 
* I haven't found a paper where people doing drug ranking (with DRL or otherwise) are taking into account constraints
  * There is a paper constraining signaling pathways but this is for better performance, not for side effects. 
    * A signaling pathway is a set of molecules in a cell that work together to control one or more cell functions, such as cell division or cell death.


#### Adding constraints

theoretical lagrange constraint version:
* get toxicity score for drugs and positions 
* do log-barrier to get penalty to add on to the loss
* do backpropagation
* clip gradients
* update the weights
* update the lagrange multiplier
  * lagrange_multiplier = lagrange_multiplier + learning_rate * constraint_violation # or something like that, eg update based on degree which constraint is being satisfied

 This formulates the constraint as an unconstrained optimization problem, and the lagrange multiplier is updated to enforce the constraint. \
 The normal PPORank will try to optimize for efficacy, lagrange will try to compromise between efficacy and toxicity. \
 The max_grad_norm is 0.5, typical grads around 0.05. Some are much higher in normal PPORank and these get clipped. \
 For lagrange, the grads will be higher but need to keep low. 
 
In training, action is often selected according to the current policy estimate. \
But to explore (and learn) other potentially more rewarding new actions, it selects an action randomly from the estimated policy distribution. \
In testing, rank the drugs by their predicted scores and select the best one.

In update phase, can either use the actions and action probs in the batch to calculate constraint penalty \
Or can run an evaluation on the policy to see what it thinks is the best drugs \
Not sure which is best


### Formulation

L = f(θ) + λg(θ) 
where f is the objective function, g is the constraint function, and λ is the Lagrange multiplier. (θ is the parameter vector)
f is the neural network for DRL, g is the toxicity function.

can say that g(θ) <= 0 is the constraint. 

g is formulated as such:
1. get average probability of each drug
2. rank in ordered list of probability of being chosen (highest prob is in theory most efficient)
3. put in ordered dictionary with drug id as key and toxicity as value
4. get penalty for each drug based on toxicity: penalty = (toxicity score)/(rank+1)
5. sum all penalties and divide by number of drugs

Mathematically:
p_i = probability of drug i
t_i = toxicity of drug i
rank_i = rank of drug i
n = number of drugs

penalty_i = t_i/(rank_i+1)
g(θ) = 1/n * Σ(penalty_i) = 1/n * Σ(t_i/(rank_i+1))

Non linear scaling version:
penalty_i = t_i * (-ln(rank_i+1)+4) - there are 38 drugs and this graph crosses the x axis around 45. This is a good way to scale the penalty to be more significant for higher toxicity drugs.


### Lagrange multiplier update

A soft constraint seems more appropriate as with a hard constraint on toxicity i could just remove all drugs with toxicity above a certain threshold. 

The lagrange multiplier is updated based on the degree to which the constraint is being satisfied: \
Δλ = lr * (target - constraint_violation) i.e. \
Δλ  = α * (c - g(θ))

Will need to clip λ to stop it becoming too large or too small.
