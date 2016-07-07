apiVersion: extensions/v1beta1
kind: Job
metadata:
  name: {user}-{experiment}-{name}
spec:
  selector:
    matchLabels:
      experiment: {experiment}
      variant: {name}
      owner: {user}
  template:
    metadata:
      name: {user}-{experiment}-{name}
      labels:
        experiment: {experiment}
        variant: {name}
        owner: {user}
    spec:
      containers:
        - name: experiment
          image: quay.io/openai/ecprice-neural-gpu
          env:
          - name: TRAIN_DIR
            value: /mnt/ecprice/neural-gpu/{session_label}/{longname}
          command:
          - bash
          - -c
          - "mkdir -p `dirname $TRAIN_DIR` && {command} --train_dir=$TRAIN_DIR"
          volumeMounts:
            - name: nvidia
              mountPath: /usr/local/nvidia
              readOnly: true
            - name: nfs
              mountPath: "/mnt"
          securityContext:
            privileged: true
          imagePullPolicy: IfNotPresent
          resources:
            requests:
              memory: "14Gi"
              cpu: 3.6
            # if you set lmits to be the same as requests
            # your QoS tier will be "Guaranteed" instead of "BestEffort"
            # so you won't get killed as easily by bursting OOM
            limits:
              memory: "14Gi"
              cpu: 3.6
      volumes:
        - name: nvidia
          hostPath:
            path: /var/lib/docker/volumes/nvidia_driver_352.63/_data
        - name: nfs
          persistentVolumeClaim:
            claimName: nfs-us-west-2a-claim
      nodeSelector:
        aws/type: g2.2xlarge
        aws/region: us-west-2
      restartPolicy: OnFailure
