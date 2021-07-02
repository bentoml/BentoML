import * as React from "react";
import { Button, Dialog, Intent, } from "@blueprintjs/core";

export interface IDeploymentDeleteConfirmProps {
  name: string;
  value: string;
  isOpen: boolean;
}

const DeploymentDeleteConfirm: React.FC<IDeploymentDeleteConfirmProps> = (props) => {
  public state: IDeploymentDeleteConfirmState = {
    isOpen: false,
  };

  const tagValue = `${props.name}:${props.value}`;
  return (
    <Dialog
      onClose={this.handleClose}
      title="Are you sure?"
      {...this.state}
    >
      <div>
        <p>
          This action cannot be undone. This will permanently delete this bento bundle and may have unintended consequences.
        </p>
      </div>
      <div>
        <div>
          <Button
            outlined="true"
            onClick=""
          >
            Delete {{ tagName }}
          </Button>
        </div>
      </div>
    </Dialog>
  );

  private handleOpen = () => this.setState({ isOpen: true });
  private handleClose = () => this.setState({ isOpen: false });
};

export default DeploymentDeleteConfirm;
