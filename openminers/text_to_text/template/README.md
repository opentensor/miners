# Template Embedding Miner

This is a template for a Text to Text Miner designed for use with the Bittensor framework and Openminers library.

The Template Text Miner provides a boilerplate for developers looking to create their own miners within the Bittensor ecosystem. It returns a simple 'Hello World!' string for all input text and serves as a starting point to develop more complex and functional mining applications.

# Running the Miner

To run the miner, navigate to the directory containing miner.py and run the following command:

```bash
python openminers/text_to_text/template/miner.py --template_arg < vaue >
```

# Customization and Implementation

You can customize and implement the template text class by modifying several methods:

- add_args: Add any new command line arguments required by your implementation.
- __init__: Initialize any necessary attributes or load models in the constructor.
- blacklist: Implement your own logic for deciding if a forward call should be blacklisted. This function should return a boolean indicating whether the call is blacklisted or not.
- priority: Adjust the priority of the forward call based on your own criteria. This function should return a float indicating the priority of the call.
- forward: The main method where the text is transformed into text completions. Your implementation should take a `List[Dict[str,str]]` of text messages and return a single string.

Please use proper type annotations when defining your functions.

Here is a basic example of how to implement these methods:

```python
class MyTextMiner( openminers.BasePromptingMiner ):
    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--my_arg', default='my_default', type=str, help='My custom argument.' )

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.my_arg = self.config.miner.my_arg
        # Load your models here

    def blacklist( self, forward_call: "bittensor.TextToTextForwardCall" ) -> Union[ Tuple[bool, str], bool ]:
        # Implement your blacklist logic here
        return False

    def priority( self, forward_call: "bittensor.TextToTextForwardCall" ) -> float:
        # Implement your priority logic here
        return 0.0

    def forward( self, text: List[Dict[str,str]] ):
        # Implement your text to completion transformation here
        return 'This is a completion for a user query!'
```

Feel free to modify this template according to your requirements and use it as a base for your new miner development.