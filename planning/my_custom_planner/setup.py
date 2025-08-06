from setuptools import find_packages, setup

package_name = 'my_custom_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ('share/' + package_name + '/launch', ['launch/my_custom_planner.launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abolfazl',
    maintainer_email='abolfazl.kabiri@aut.ac.ir',
    description='A custom planner node for Autoware',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_custom_planner_node = my_custom_planner.planner_node:main' ,
        ],
    },
)
