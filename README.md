### Environment Setting

If using docker, please read and run docker build from the docker file given in /dockerfiles directory

For trouble shooting please refer to this habitat notes: https://docs.google.com/document/d/1NEJ0RqCSZRNocpizpMCmUxl1wZGfQBKppSHvmvmRUJU/edit


### Training for fetch robot
```python habitat-baselines/habitat_baselines/run.py --config-name=social_nav/social_nav_fetch.yaml``

To resume training:
```python habitat-baselines/habitat_baselines/run.py --config-name=social_nav/social_nav_fetch.yaml habitat_baselines.load_resume_state_config=True```

To evaluate:
```python habitat-baselines/habitat_baselines/run.py --config-name=social_nav/social_nav_fetch.yaml habitat_baselines.evaluate=True habitat_baselines.eval_ckpt_path_dir=/habitat-lab/data/checkpoints/latest.pth habitat_baselines.eval.should_load_ckpt=True```


### Dataset Downloads

Dataset Download
```python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path /path/to/data/```
Dataset uids are listed here:
https://github.com/facebookresearch/habitat-sim/blob/v0.3.0/src_python/habitat_sim/utils/datasets_download.py

To run the controller in habitat_sim, follow instructions:

1. test if glxgears working well
2. install habitat-sim if dockerfile haven't install
    ```conda install habitat-sim withbullet -c conda-forge -c aihabitat```
3. download dataset
    ```python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /path/to/data/```
    example dataset:
    ```python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path /path/to/data/```

4. Interactive Testing
    ```bash
    #C++
    #./build/viewer if compiling locally
    habitat-viewer /path/to/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb

    #Python
    #NOTE: depending on your choice of installation, you may need to add '/path/to/habitat-sim' to your PYTHONPATH.
    #e.g. from 'habitat-sim/' directory run 'export PYTHONPATH=$(pwd)'
    python examples/viewer.py --scene /path/to/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
    ```

