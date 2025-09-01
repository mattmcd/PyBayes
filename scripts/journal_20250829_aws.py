# %%
import boto3
import requests
from botocore.exceptions import ClientError
import os
import subprocess
from pathlib import Path

# %%
def auth_current_ip(security_group, ip):
    try:
        security_group.authorize_ingress(
            DryRun=False,
            IpPermissions=[
                {
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpProtocol': 'tcp',
                    'IpRanges': [
                        {
                            'CidrIp': f'{ip}/32',
                        }
                    ]
                }
            ]
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidPermission.Duplicate':
            print('Rule already exists')
        else:
            raise e


# %%
def revoke_old_ips(security_group, new_ip):
    ip_permissions = security_group.ip_permissions
    old_ips = [
        r['CidrIp'] for r in ip_permissions[0]['IpRanges'] if new_ip not in r['CidrIp']
    ]
    # Revoke access for all other IP addresses
    for old_ip in old_ips:
        security_group.revoke_ingress(
            CidrIp=old_ip,
            FromPort=22,
            ToPort=22,
            IpProtocol='tcp',
        )
    print(f'Revoked access for {len(old_ips)} old IP addresses')


# %%
class SecurityGroupIpUpdater:
    def __init__(self, security_group=None, ip=None):
        self.security_group = security_group
        self.ip = ip

    def update(self):
        auth_current_ip(self.security_group, self.ip)
        revoke_old_ips(self.security_group, self.ip)

    @classmethod
    def from_env(cls):
        return cls(
            security_group=boto3.resource('ec2').SecurityGroup(os.environ['AWS_SECURITY_GROUP_ID']),
            ip=requests.get('https://checkip.amazonaws.com').text.strip()
        )


# %%
class InstanceConnection:
    def __init__(self, instance, key_file):
        self.instance = instance
        self.key_file = key_file

    def start(self):
        self.instance.start()

    def stop(self):
        self.instance.stop()

    @property
    def is_running(self):
        return self.instance.state['Name'] == 'running'

    def tunnel(self):
        ssh_cmd = (
            f'ssh -N -L 3128:localhost:3128 -i "{self.key_file}" '
            f'ubuntu@{self.instance.public_dns_name}'
        )
        # Still fiddling with this to keep tunnel open rather than just
        # executing tunnel command and then immediately closing it when subprocess exits.
        return subprocess.run(ssh_cmd.split(' '), capture_output=True)

    @classmethod
    def from_env(cls):
        ec2 = boto3.resource('ec2')
        instance = ec2.Instance(os.environ['AWS_INSTANCE_ID'])
        key_file = (Path.home() / os.environ['AWS_KEY_FILE']).resolve().as_posix()
        return cls(instance, key_file)


# %%
if __name__ == '__main__':
    updater = SecurityGroupIpUpdater.from_env()
    updater.update()
    conn = InstanceConnection.from_env()
