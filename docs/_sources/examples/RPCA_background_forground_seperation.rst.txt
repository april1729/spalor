RPCA for Background-Foreground Seperation
##############

.. code:: ipython3

    import sys
    sys.path.append('../')
    from datasets.data_loader import *
    from RPCA.algorithms import *
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    v=load_video("../datasets/videos/escalator.avi")
    (n_frames,d1,d2)=v.shape
    v=v.reshape(n_frames, d1*d2)
    


.. code:: ipython3

    (L,S)=altProjNiave(v, 2,100*n_frames)

.. code:: ipython3

    L=L.reshape(n_frames,d1,d2)
    S=S.reshape(n_frames,d1,d2)
    v=v.reshape(n_frames,d1,d2)
    all=np.concatenate((v,L,S), axis=2)
    
    plt.imshow(all[1,:,:])
    plt.show()



.. image:: RPCA_background_forground_seperation_files/RPCA_background_forground_seperation_2_0.png


.. code:: ipython3

    fig = plt.figure()
    ims = []
    for i in range(n_frames):
        im = plt.imshow(all[i,:,:], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat_delay=10)
    plt.show()
