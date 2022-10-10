# Generated from /home/mattmcd/Work/Projects/PyBayes/scripts/StateMachine.g4 by ANTLR 4.10.1
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,0,7,50,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,
        6,7,6,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,2,1,2,1,2,1,
        2,1,2,1,2,1,3,1,3,1,3,1,4,1,4,1,4,1,5,4,5,40,8,5,11,5,12,5,41,1,
        6,4,6,45,8,6,11,6,12,6,46,1,6,1,6,0,0,7,1,1,3,2,5,3,7,4,9,5,11,6,
        13,7,1,0,4,1,0,97,122,1,0,48,57,3,0,65,90,95,95,97,122,2,0,9,10,
        32,32,51,0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,0,0,0,0,7,1,0,0,0,0,9,1,0,
        0,0,0,11,1,0,0,0,0,13,1,0,0,0,1,15,1,0,0,0,3,22,1,0,0,0,5,26,1,0,
        0,0,7,32,1,0,0,0,9,35,1,0,0,0,11,39,1,0,0,0,13,44,1,0,0,0,15,16,
        5,101,0,0,16,17,5,118,0,0,17,18,5,101,0,0,18,19,5,110,0,0,19,20,
        5,116,0,0,20,21,5,115,0,0,21,2,1,0,0,0,22,23,5,101,0,0,23,24,5,110,
        0,0,24,25,5,100,0,0,25,4,1,0,0,0,26,27,5,115,0,0,27,28,5,116,0,0,
        28,29,5,97,0,0,29,30,5,116,0,0,30,31,5,101,0,0,31,6,1,0,0,0,32,33,
        5,61,0,0,33,34,5,62,0,0,34,8,1,0,0,0,35,36,7,0,0,0,36,37,7,1,0,0,
        37,10,1,0,0,0,38,40,7,2,0,0,39,38,1,0,0,0,40,41,1,0,0,0,41,39,1,
        0,0,0,41,42,1,0,0,0,42,12,1,0,0,0,43,45,7,3,0,0,44,43,1,0,0,0,45,
        46,1,0,0,0,46,44,1,0,0,0,46,47,1,0,0,0,47,48,1,0,0,0,48,49,6,6,0,
        0,49,14,1,0,0,0,3,0,41,46,1,6,0,0
    ]

class StateMachineLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    CODE = 5
    NAME = 6
    WS = 7

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'events'", "'end'", "'state'", "'=>'" ]

    symbolicNames = [ "<INVALID>",
            "CODE", "NAME", "WS" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "CODE", "NAME", "WS" ]

    grammarFileName = "StateMachine.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


