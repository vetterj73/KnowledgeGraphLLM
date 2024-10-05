az stack sub create --deny-settings-mode denyWriteAndDelete `
    -n mercury-knowledge `
    --location centralus `
    --action-on-unmanage deleteAll `
    --yes `
    --subscription cfef89fb-d471-42aa-9457-57a8778d3638 `
    -p main.bicepparam `
    -f main.bicep `
	--deny-settings-excluded-actions "Microsoft.Web/sites/publishxml/action Microsoft.Web/sites/config/list/action" `
	--ep '82a13213-1581-4c8d-a922-6aea478a02f1'