from dataclasses import dataclass, field
from state_machine.StateMachineLexer import StateMachineLexer
from state_machine.StateMachineParser import StateMachineParser
from state_machine.StateMachineListener import StateMachineListener
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker


@dataclass
class Event:
    code: str
    name: str

@dataclass
class State:
    code: str
    name: str
    transitions: dict = field(default_factory=dict)

    def __add__(self, event: Event):
        return TransitionFactory(self, event)


@dataclass
class Transition:
    source: State
    dest:  State
    event: Event

    def __post_init__(self):
        self.source.transitions[self.event.code] = self.dest.code


class TransitionFactory:
    def __init__(self, state: State, event: Event):
        self.state = state
        self.event = event

    def __rshift__(self, new_state):
        return Transition(self.state, new_state, self.event)


class StateMachine:
    def __init__(self):
        self.states = dict()
        self.events = dict()
        self.transitions = dict()
        self.start = None
        self.reset_events = set()

    def set_start(self, state_code):
        # Starting state to return to after reset events
        self.start = state_code
        return self

    def add_reset_event(self, event_code):
        # Set of events that cause a reset of the state to starting state
        self.reset_events.add(event_code)
        return self

    def add_state(self, code, name):
        self.states[code] = State(code, name)
        return self

    def add_event(self, code, name):
        self.events[code] = Event(code, name)
        return self

    def add_transition(self, transition: str):
        """Use the state and envent codes to define a transition from state to to state in response to event

        :param transition: string 'src event dest'
        :return:
        """
        src, event, dest = transition.split(' ')
        self.transitions[transition] = Transition(self.states[src], self.states[dest], self.events[event] )
        return self

    def __repr__(self):
        out_str = ''
        out_str += 'events\n'
        for event in self.events.values():
            out_str += f'  {event.code} {event.name}\n'
        out_str += 'end\n\n'

        for state in self.states.values():
            out_str += f'state {state.name}\n'
            for event_code, dest_state_code in state.transitions.items():
                out_str += f'  {self.events[event_code].name} => {self.states[dest_state_code].name}\n'
            out_str += 'end\n\n'
        return out_str

    @classmethod
    def from_text(cls, txt):
        return build_state_machine(txt)


def example_state_machine():
    sm = StateMachine()
    sm.add_state(
        's0', 'start'
    ).add_state(
        's1', 'finish'
    ).add_event(
        'e0', 'move'
    ).add_transition(
        's0 e0 s1'
    ).set_start('s0')
    return sm


def example_gothic_controller():
    sm = StateMachine()
    states = 'idle active waiting_for_light waiting_for_drawer unlocked_panel'.split(' ')
    for i, state in enumerate(states):
        sm.add_state(f's{i}', state)
    events = 'door_closed drawer_opened light_on door_opened panel_closed'.split(' ')
    for i, event in enumerate(events):
        sm.add_event(f'e{i}', event)
    # Transitions from state and event codes is a bit ugly - refactor to state and event names?
    sm.add_transition('s0 e0 s1')
    sm.add_transition('s1 e1 s2')
    sm.add_transition('s1 e2 s3')
    sm.add_transition('s2 e2 s4')
    sm.add_transition('s3 e1 s4')
    sm.add_transition('s4 e4 s0')
    sm.set_start('s0')
    sm.add_reset_event('e3')
    return sm


def parse_state_machine(machine_desc):
    input_stream = InputStream(machine_desc)
    lexer = StateMachineLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = StateMachineParser(stream)
    tree = parser.file_()
    return tree


def build_state_machine(machine_desc):
    tree = parse_state_machine(machine_desc)
    builder = StateMachineBuilder()
    walker = ParseTreeWalker()
    walker.walk(builder, tree)
    return builder.state_machine


class StateMachineBuilder(StateMachineListener):
    def __init__(self):
        self.states = dict()
        self.events = dict()
        self.transitions = []
        self.state_machine = StateMachine()

    def enterEvent(self, ctx:StateMachineParser.EventContext):
        event = Event(ctx.CODE().getText(), ctx.NAME().getText())
        self.events[event.code] = event
        self.state_machine.add_event(event.code, event.name)
        # print(f'Entering event {event.name}')

    def enterMachine_state(self, ctx:StateMachineParser.Machine_stateContext):
        state = State(f's{len(self.states)}', ctx.NAME().getText())
        self.states[state.code] = state
        self.state_machine.add_state(state.code, state.name)
        # print(f'Entering state {state.name}')

    def enterTransition(self, ctx:StateMachineParser.TransitionContext):
        source_state = ctx.parentCtx.NAME().getText()
        dest_state = ctx.NAME(1).getText()
        event_name = ctx.NAME(0).getText()
        self.transitions.append( (source_state, dest_state, event_name))

    def exitFile(self, ctx:StateMachineParser.FileContext):
        # print(self.transitions)
        # Construct the state machine transitions
        state_lookup = {state.name: state.code for state in self.states.values()}
        event_lookup = {event.name: event.code for event in self.events.values()}
        for source_state, dest_state, event_name in self.transitions:
            self.state_machine.add_transition(
                f'{state_lookup[source_state]} {event_lookup[event_name]} {state_lookup[dest_state]}'
            )


