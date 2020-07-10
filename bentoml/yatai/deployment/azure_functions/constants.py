MAX_RESOURCE_GROUP_NAME_LENGTH = 90
MAX_STORAGE_ACCOUNT_NAME_LENGTH = 24
MAX_FUNCTION_NAME_LENGTH = 60
MAX_CONTAINER_REGISTRY_NAME_LENGTH = 50

# https://docs.microsoft.com/en-us/azure/azure-functions/functions-premium-plan
AZURE_FUNCTIONS_PREMIUM_PLAN_SKUS = ['EP1', 'EP2', 'EP3']
# https://docs.microsoft.com/en-us/java/api/com.microsoft.azure.functions.annotation.httptrigger.authlevel?view=azure-java-stable
AZURE_FUNCTIONS_AUTH_LEVELS = ['anonymous', 'function', 'admin']

DEFAULT_MIN_INSTANCE_COUNT = 1
DEFAULT_MAX_BURST = 20
DEFAULT_PREMIUM_PLAN_SKU = AZURE_FUNCTIONS_PREMIUM_PLAN_SKUS[0]
DEFAULT_FUNCTION_AUTH_LEVEL = AZURE_FUNCTIONS_AUTH_LEVELS[0]
