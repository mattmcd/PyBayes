# %%
import boto3
import requests
from botocore.exceptions import ClientError
import os

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
if __name__ == '__main__':
    updater = SecurityGroupIpUpdater.from_env()
    updater.update()