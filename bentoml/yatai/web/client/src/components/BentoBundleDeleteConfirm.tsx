import * as React from "react";
import { Button, Dialog, Intent, } from "@blueprintjs/core";
import { template } from "lodash";

export interface IBentoBundleDeleteConfirmProps {
  name: string;
  value: string;
  isOpen: boolean;
}

const BentoBundleDeleteConfirm: React.FC<IBentoBundleDeleteConfirmProps> = (props) => {
  let isOpen = false;

  const tagValue = `${props.name}:${props.value}`;
  return (
    <div>
      <Button onClick={handleOpen}>Delete</Button>
    <Dialog
      onClose={handleClose}
      title="Are you sure?"
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
    </div>
  );

  const handleClose = () => !isOpen;
};

export default BentoBundleDeleteConfirm;
