import simpy


class Source:
    """2024-02-11 Simple Source in a Game Loop
    Stock level increases by 1 in each timestep, then decreases when used by a consuming Sink
    """
    def __init__(self, env, name=''):
        self.env = env
        self.action = env.process(self.run())
        self.pipe = simpy.Container(self.env)
        self.name = name

    def run(self):
        while True:
            self.pipe.put(1)
            print(f'Source{self.name} called at {self.env.now}.  Stock: {self.stock}')
            yield self.env.timeout(1)

    @property
    def stock(self):
        return self.pipe.level

    def get(self, amount):
        return self.pipe.get(amount)


class Sink:
    """2024-02-11 Simple Sink in a Game Loop
    Waits until it can request _amount_ units from Sink
    """
    def __init__(self, env, source, amount=5):
        self.env = env
        self.source = source
        self.action = env.process(self.run())
        self.stock = 0
        self.amount = amount

    def run(self):
        while True:
            yield self.source.get(self.amount)
            self.stock += self.amount
            print(f'Sink called at {self.env.now}.  Sink stock {self.stock}.  Source stock {self.source.stock}')


class Converter(Sink):
    def __init__(self, env, source, amount=5, convert_amount=1):
        super().__init__(env, source, amount)
        self.convert_amount = convert_amount

    def run(self):
        while True:
            yield self.source.get(self.amount)
            self.stock += self.convert_amount
            print(f'Converted {self.amount} to {self.convert_amount} at {self.env.now}.  Sink stock {self.stock}.  Source stock {self.source.stock}')


class MultiConverter:
    def __init__(self, env, sources, amounts):
        self.env = env
        self.sources = sources
        self.amounts = amounts
        self.action = env.process(self.run())
        self.stock = 0

    def run(self):
        while True:
            # Hardwire to two sources for the moment
            # Not quite working: should only fire when both queues have enough resources
            # but currently removes stock from one queue at a time
            # yield (self.sources[0].stock >= self.amounts[0]) & (self.sources[1].stock  > self.amounts[1])
            yield self.sources[0].get(self.amounts[0]) & self.sources[1].get(self.amounts[1])
            self.stock += 1
            print(f'Sink called at {self.env.now}.  Sink stock {self.stock}.')