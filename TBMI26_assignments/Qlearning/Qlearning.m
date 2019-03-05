%% Initialization
%  Initialize the world, Q-table, and hyperparameters

global GWTERM;
[row, col, ~] = find(GWTERM == 1);
gwinit(1);
gwdraw
Q = abs(randn(state.ysize,state.xsize,4));
gamma = 0.9;
lr = 0.2;
epsilon = 0.3;
%reward = zeros(state.xsize*state.ysize,4);
% reward = zeros(state.ysize,state.xsize,4);
% reward(row+1,col,1) = 1; %Up
% reward(row-1,col,2) = 1; %Down
% reward(row,col+1,3) = 1; %Left
% reward(row,col-1,4) = 1; %Right

%% Training loop
%  Train the agent using the Q-learning algorithm.
for i = 1:100
state = gwstate;
    while(state.isterminal ~= 1)
        a = chooseaction(Q,state.pos(1),state.pos(2),[1 2 3 4], [1 1 1 1], epsilon);
        %r = reward(state.pos(1),state.pos(2),a);
        r = state.feedback;
        next_s = gwaction(a);
        q = max(Q,[],3);
        Q(state.pos(1),state.pos(2),a) = (1-lr)*Q(state.pos(1),state.pos(2),a) + ...
        lr*(r + gamma*q(next_s.pos(1),next_s.pos(2)));
        state = next_s;
    end
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0, always pick
%  the optimal action.


