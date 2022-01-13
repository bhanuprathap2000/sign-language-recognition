 pose = (
            np.array(list(map(np.array, df.pose.values)))
            .astype(np.float32)
        )
        #
        pose_shape=pose.shape
        total_frame=pose_shape[0]
        temp_total_shape=total_frame

        while (temp_total_shape*pose_shape[1]*pose_shape[2])%50!=0:
            temp_total_shape+=1
        print(temp_total_shape-total_frame)

        pose=np.concatenate([pose,np.zeros((temp_total_shape-total_frame,33,2))])
        #
        pose=pose.reshape(-1,50)

        h1 = (
            np.array(list(map(np.array, df.hand1.values)))
            .astype(np.float32)
        )
        h1=np.concatenate([h1,np.zeros((pose.shape[0]-total_frame,21,2))])
        h1=h1.reshape(-1,42)
        h2 = (
            np.array(list(map(np.array, df.hand2.values)))
            .astype(np.float32)
        )
        h2=np.concatenate([h2,np.zeros((pose.shape[0]-total_frame,21,2))])
        h2=h2.reshape(-1,42)
        print(pose.shape)
        print(h1.shape,h2.shape)
        # sys.exit(1)