# Generated from /home/mattmcd/Work/Projects/PyBayes/scripts/StateMachine.g4 by ANTLR 4.10.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .StateMachineParser import StateMachineParser
else:
    from StateMachineParser import StateMachineParser

# This class defines a complete listener for a parse tree produced by StateMachineParser.
class StateMachineListener(ParseTreeListener):

    # Enter a parse tree produced by StateMachineParser#file.
    def enterFile(self, ctx:StateMachineParser.FileContext):
        pass

    # Exit a parse tree produced by StateMachineParser#file.
    def exitFile(self, ctx:StateMachineParser.FileContext):
        pass


    # Enter a parse tree produced by StateMachineParser#events.
    def enterEvents(self, ctx:StateMachineParser.EventsContext):
        pass

    # Exit a parse tree produced by StateMachineParser#events.
    def exitEvents(self, ctx:StateMachineParser.EventsContext):
        pass


    # Enter a parse tree produced by StateMachineParser#event.
    def enterEvent(self, ctx:StateMachineParser.EventContext):
        pass

    # Exit a parse tree produced by StateMachineParser#event.
    def exitEvent(self, ctx:StateMachineParser.EventContext):
        pass


    # Enter a parse tree produced by StateMachineParser#machine_state.
    def enterMachine_state(self, ctx:StateMachineParser.Machine_stateContext):
        pass

    # Exit a parse tree produced by StateMachineParser#machine_state.
    def exitMachine_state(self, ctx:StateMachineParser.Machine_stateContext):
        pass


    # Enter a parse tree produced by StateMachineParser#transition.
    def enterTransition(self, ctx:StateMachineParser.TransitionContext):
        pass

    # Exit a parse tree produced by StateMachineParser#transition.
    def exitTransition(self, ctx:StateMachineParser.TransitionContext):
        pass



del StateMachineParser