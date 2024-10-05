param location string
param tags object

resource web 'Microsoft.Web/sites@2022-03-01' = {
  name: 'mercuryknowledge-back'
  location: location
  tags: union(tags, { 'azd-service-name': 'web' })
  kind: 'app,linux'
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      alwaysOn: true
      linuxFxVersion: 'PYTHON|3.12'
      ftpsState: 'FtpsOnly'
      appCommandLine: 'entrypoint.sh'
      minTlsVersion: '1.2'
    }
    httpsOnly: true
  }
  identity: {
    type: 'SystemAssigned'
  }
  
  resource appSettings 'config' = {
    name: 'appsettings'
    properties: {
      SCM_DO_BUILD_DURING_DEPLOYMENT: '1'
      FILE_ENCODING: 'ISO-8859-1'
      GRAPH_PATH: 'call_log_graph.gpickle'
      INDEX_PATH: 'call_log_index.bin'
      MAPPING_PATH: 'call_log_id_to_entity.pkl'
      OPEN_API_KEY: ''
      SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL: 'true'
      WEBSITE_HTTPLOGGING_RETENTION_DAYS: '3'
    }
  }
}


resource appServicePlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: 'jvetter_asp_6621'
  location: location
  tags: tags
  sku: {
    name: 'B1'
  }
  properties: {
    reserved: true
  }
}

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2020-03-01-preview' = {
  name: 'logmercuryknowledge'
  location: location
  tags: tags
  properties: any({
    retentionInDays: 30
    features: {
      searchVersion: 1
    }
    sku: {
      name: 'PerGB2018'
    }
  })
}

module applicationInsightsResources 'appinsights.bicep' = {
  name: 'applicationinsights-resources'
  params: {
    location: location
    tags: tags
    workspaceId: logAnalyticsWorkspace.id
  }
}

output WEB_URI string = 'https://${web.properties.defaultHostName}'
output APPLICATIONINSIGHTS_CONNECTION_STRING string = applicationInsightsResources.outputs.APPLICATIONINSIGHTS_CONNECTION_STRING

resource webAppSettings 'Microsoft.Web/sites/config@2022-03-01' existing = {
  name: web::appSettings.name
  parent: web
}

var webAppSettingsKeys = map(items(webAppSettings.list().properties), setting => setting.key)
output WEB_APP_SETTINGS array = webAppSettingsKeys
output WEB_APP_LOG_STREAM string = format('https://portal.azure.com/#@/resource{0}/logStream', web.id)
output WEB_APP_SSH string = format('https://{0}.scm.azurewebsites.net/webssh/host', web.name)
output WEB_APP_CONFIG string = format('https://portal.azure.com/#@/resource{0}/configuration', web.id)
