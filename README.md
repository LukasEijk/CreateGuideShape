# CreateGuideShape
Takes mathematical input functions and creates a 3D object file from it.

Using the Jupyter notebook, one can easily create a guide shape that can then be imported into other programs such as McStas.
McStas can then be used to simulate neutron scattering experiments by using the created guide shapes as parts in the simulation in the form of neutron guides. 

The Jupyter notebook generates the file try_ges.obj. This generated file can then be imported into Blender for visualization. 
A png file is also added to show the visualized result in Blender.

Note that the guide structure is sliced in such a way that as few shapes as possible are created, but as many as necessary, so that the course of the input function is preserved as far as possible. A maximum length of the sections can be defined.
