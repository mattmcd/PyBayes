# Generated from /home/mattmcd/Work/Projects/PyBayes/scripts/StateMachine.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,7,41,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,1,0,1,0,4,0,13,
        8,0,11,0,12,0,14,1,1,1,1,4,1,19,8,1,11,1,12,1,20,1,1,1,1,1,2,1,2,
        1,2,1,3,1,3,1,3,4,3,31,8,3,11,3,12,3,32,1,3,1,3,1,4,1,4,1,4,1,4,
        1,4,0,0,5,0,2,4,6,8,0,0,38,0,10,1,0,0,0,2,16,1,0,0,0,4,24,1,0,0,
        0,6,27,1,0,0,0,8,36,1,0,0,0,10,12,3,2,1,0,11,13,3,6,3,0,12,11,1,
        0,0,0,13,14,1,0,0,0,14,12,1,0,0,0,14,15,1,0,0,0,15,1,1,0,0,0,16,
        18,5,1,0,0,17,19,3,4,2,0,18,17,1,0,0,0,19,20,1,0,0,0,20,18,1,0,0,
        0,20,21,1,0,0,0,21,22,1,0,0,0,22,23,5,2,0,0,23,3,1,0,0,0,24,25,5,
        5,0,0,25,26,5,6,0,0,26,5,1,0,0,0,27,28,5,3,0,0,28,30,5,6,0,0,29,
        31,3,8,4,0,30,29,1,0,0,0,31,32,1,0,0,0,32,30,1,0,0,0,32,33,1,0,0,
        0,33,34,1,0,0,0,34,35,5,2,0,0,35,7,1,0,0,0,36,37,5,6,0,0,37,38,5,
        4,0,0,38,39,5,6,0,0,39,9,1,0,0,0,3,14,20,32
    ]

class StateMachineParser ( Parser ):

    grammarFileName = "StateMachine.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'events'", "'end'", "'state'", "'=>'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "CODE", "NAME", "WS" ]

    RULE_file = 0
    RULE_events = 1
    RULE_event = 2
    RULE_machine_state = 3
    RULE_transition = 4

    ruleNames =  [ "file", "events", "event", "machine_state", "transition" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    CODE=5
    NAME=6
    WS=7

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class FileContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def events(self):
            return self.getTypedRuleContext(StateMachineParser.EventsContext,0)


        def machine_state(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(StateMachineParser.Machine_stateContext)
            else:
                return self.getTypedRuleContext(StateMachineParser.Machine_stateContext,i)


        def getRuleIndex(self):
            return StateMachineParser.RULE_file

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFile" ):
                listener.enterFile(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFile" ):
                listener.exitFile(self)




    def file_(self):

        localctx = StateMachineParser.FileContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_file)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 10
            self.events()
            self.state = 12 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 11
                self.machine_state()
                self.state = 14 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==StateMachineParser.T__2):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EventsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def event(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(StateMachineParser.EventContext)
            else:
                return self.getTypedRuleContext(StateMachineParser.EventContext,i)


        def getRuleIndex(self):
            return StateMachineParser.RULE_events

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEvents" ):
                listener.enterEvents(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEvents" ):
                listener.exitEvents(self)




    def events(self):

        localctx = StateMachineParser.EventsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_events)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 16
            self.match(StateMachineParser.T__0)
            self.state = 18 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 17
                self.event()
                self.state = 20 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==StateMachineParser.CODE):
                    break

            self.state = 22
            self.match(StateMachineParser.T__1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EventContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CODE(self):
            return self.getToken(StateMachineParser.CODE, 0)

        def NAME(self):
            return self.getToken(StateMachineParser.NAME, 0)

        def getRuleIndex(self):
            return StateMachineParser.RULE_event

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEvent" ):
                listener.enterEvent(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEvent" ):
                listener.exitEvent(self)




    def event(self):

        localctx = StateMachineParser.EventContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_event)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 24
            self.match(StateMachineParser.CODE)
            self.state = 25
            self.match(StateMachineParser.NAME)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Machine_stateContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NAME(self):
            return self.getToken(StateMachineParser.NAME, 0)

        def transition(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(StateMachineParser.TransitionContext)
            else:
                return self.getTypedRuleContext(StateMachineParser.TransitionContext,i)


        def getRuleIndex(self):
            return StateMachineParser.RULE_machine_state

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMachine_state" ):
                listener.enterMachine_state(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMachine_state" ):
                listener.exitMachine_state(self)




    def machine_state(self):

        localctx = StateMachineParser.Machine_stateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_machine_state)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 27
            self.match(StateMachineParser.T__2)
            self.state = 28
            self.match(StateMachineParser.NAME)
            self.state = 30 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 29
                self.transition()
                self.state = 32 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==StateMachineParser.NAME):
                    break

            self.state = 34
            self.match(StateMachineParser.T__1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TransitionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NAME(self, i:int=None):
            if i is None:
                return self.getTokens(StateMachineParser.NAME)
            else:
                return self.getToken(StateMachineParser.NAME, i)

        def getRuleIndex(self):
            return StateMachineParser.RULE_transition

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransition" ):
                listener.enterTransition(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransition" ):
                listener.exitTransition(self)




    def transition(self):

        localctx = StateMachineParser.TransitionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_transition)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 36
            self.match(StateMachineParser.NAME)
            self.state = 37
            self.match(StateMachineParser.T__3)
            self.state = 38
            self.match(StateMachineParser.NAME)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





