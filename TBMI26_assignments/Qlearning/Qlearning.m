    %% Initialization
%  Initialize the world, Q-table, and hyperparameters

global GWTERM;
[row, col, ~] = find(GWTERM == 1);
MAP = 3;
gwinit(MAP);
state = gwstate;
Q = rand(state.ysize,state.xsize,4);
gamma = 0.9;
lr = 0.3;
epsilon = 0.9;

%% Training loop
%  Train the agent using the Q-learning algorithm.
Episodes = 1000;
for i = 1:Episodes
gwinit(MAP);
state = gwstate;
if(i/Episodes == 0.4)
    epsilon = 0.3;
end
if(mod(i,100) == 0)
    i
end
    while(state.isterminal ~= 1)
        b = chooseaction(Q,state.pos(1),state.pos(2),[1 2 3 4], [0.25 0.25 0.25 0.25], epsilon);
        [next_s,a] = gwaction(b);
        r = next_s.feedback;
        if(next_s.isvalid)
            q = max(Q,[],3);
            Q(state.pos(1),state.pos(2),b) = (1-lr)*Q(state.pos(1),state.pos(2),b) + ...
            lr*(r + gamma*q(next_s.pos(1),next_s.pos(2)));
        else
            Q(state.pos(1),state.pos(2),a) = -inf;
        end
        state = next_s;
    end
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0, always pick
%  the optimal action.

P = gwgetpolicy(Q);
gwinit(MAP);
s = gwstate;
while(s.isterminal ~= 1)
    a = P(s.pos(1),s.pos(2));
    s = gwaction(a);
    gwdraw(Q)
end
