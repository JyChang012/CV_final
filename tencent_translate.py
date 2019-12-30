from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tmt.v20180321 import tmt_client, models
import time

secretId = "AKIDqQknaVT4g6GExtgalhNqtkLgnnWz7OWP"
secretKey = "lGUR8ZjcEApGQJvGJ5aonxpQlpc9QQDs"

cred = credential.Credential(secretId, secretKey)
httpProfile = HttpProfile()
httpProfile.endpoint = "tmt.tencentcloudapi.com"
clientProfile = ClientProfile()
clientProfile.httpProfile = httpProfile
client = tmt_client.TmtClient(cred, "ap-guangzhou", clientProfile)


def translate(source_text: str, source='en', target='zh', project_id=0):
    try:
        req = models.TextTranslateRequest()
        req.Source = source
        req.SourceText = source_text
        req.Target = target
        req.ProjectId = project_id
        resp = client.TextTranslate(req)
        return resp.TargetText
    except TencentCloudSDKException as err:
        if err.get_code() == 'FailedOperation.NoFreeAmount':
            raise err
        else:
            time.sleep(.1)  # Deal with tencent cloud request freq limit
            return translate(source_text, source, target)


if __name__ == '__main__':
    print(translate('hello'))
