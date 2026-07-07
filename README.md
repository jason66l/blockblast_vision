# blockblast_vision

Built a real-time solver for the mobile game Block Blast using computer vision and algorithmic
optimization.

Mirrored iPhone screen onto my computer and wrote a Python script that detects the game
board and available pieces directly from the live screen feed. Using OpenCV, the program
identifies piece shapes, converts them into matrix representations, and reconstructs the current
board state dynamically.

Once the board and pieces are converted into matrices, the system applies an optimization
algorithm (adapted and improved from open-source research) to evaluate all possible
placements and select the most optimal move based on survival and scoring heuristics.
