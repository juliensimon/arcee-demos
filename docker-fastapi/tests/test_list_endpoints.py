from requests import request

def invoke(url="http://localhost:80", path="/", method="GET",
           headers={"Content-Type": "application/json"}, body=None,
           timeout=60):
    return request(method, f"{url}{path}", headers=headers, data=body, timeout=timeout)

def test_list_endpoints():
    response = invoke(path="/list_endpoints", method="GET")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    for endpoint in data:
        assert 'EndpointName' in endpoint
        assert isinstance(endpoint['EndpointName'], str)
        
        assert 'InstanceType' in endpoint
        assert isinstance(endpoint['InstanceType'], str)
        
        assert 'Container' in endpoint
        assert isinstance(endpoint['Container'], str)
        
        assert 'ModelEnvironment' in endpoint
        assert isinstance(endpoint['ModelEnvironment'], dict)

