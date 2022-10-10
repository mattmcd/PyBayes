grammar StateMachine;

file: events machine_state+;

events: 'events' event+ 'end';

event: CODE NAME;

machine_state: 'state' NAME transition* 'end';

transition: NAME '=>' NAME;

CODE: [a-z][0-9];

NAME: [A-Za-z_]+;

WS: [\t\n ]+ -> skip;