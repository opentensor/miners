# Template Embedding Miner

This is a template for an Embedding Miner designed for use with the Bittensor framework and Openminers library.

The Template Embedding Miner provides a boilerplate for developers looking to create their own miners within the Bittensor ecosystem. It returns zero embeddings for all input text and serves as a starting point to develop more complex and functional mining applications.

# Running the Miner

To run the miner, navigate to the directory containing TemplateEmbeddingMiner.py and run the following command:

```bash
python openminers/text_to_embedding/template/TemplateEmbeddingMiner.py --template_arg < vaue >
```

# Customization and Implementation

You can customize and implement the TemplateEmbeddingMiner class by modifying several methods:

- add_args: Add any new command line arguments required by your implementation.
- __init__: Initialize any necessary attributes or load models in the constructor.
- blacklist: Implement your own logic for deciding if a forward call should be blacklisted. This function should return a boolean indicating whether the call is blacklisted or not.
- priority: Adjust the priority of the forward call based on your own criteria. This function should return a float indicating the priority of the call.
- forward: The main method where the text is transformed into embeddings. Your implementation should take a list of text strings and return a tensor of embeddings.

Please use proper type annotations when defining your functions.

Here is a basic example of how to implement these methods:

```python
class MyEmbeddingMiner( openminers.BaseEmbeddingMiner ):
    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--my_arg', default='my_default', type=str, help='My custom argument.' )

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.my_arg = self.config.miner.my_arg
        # Load your models here

    def blacklist( self, forward_call: "bittensor.TextToEmbeddingForwardCall" ) -> Union[ Tuple[bool, str], bool ]:
        # Implement your blacklist logic here
        return False

    def priority( self, forward_call: "bittensor.TextToEmbeddingForwardCall" ) -> float:
        # Implement your priority logic here
        return 0.0

    def forward( self, text: List[str] ) -> torch.Tensor:
        # Implement your text to embedding transformation here
        return torch.zeros( ( self.config.miner.embedding_size ) )
```

Feel free to modify this template according to your requirements and use it as a base for your new miner development.