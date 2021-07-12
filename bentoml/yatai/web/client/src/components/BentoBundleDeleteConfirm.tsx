import * as React from "react";
import { Button, AnchorButton, Dialog, Intent, Tooltip } from "@blueprintjs/core";
import { YataiToaster } from "../utils/Toaster";

export interface IBentoBundleDeleteConfirmationProps {
  name: string;
  value: string;
  isOpen: boolean;
}

const handleDelete = async (bundle) => {
  // const response = await delete;
  // const message = response.success ? "has been deleted." : `has not been deleted.\nError encountered: ${response.error}`;
  // const toastState = {
  //   message: `${bundle} ${message}`,
  //   intent: props.success ? Intent.SUCCESS: Intent.DANGER,
  // };
  // YataiToaster.show({toastState})
};

const BentoBundleDeleteConfirmation: React.FC<IBentoBundleDeleteConfirmationProps> = (
  props
) => {
  const [open, setOpen] = React.useState(false);

  // let isOpen = false;
  const bundle = `${props.name}:${props.value}`;

  return (
    <div>
      <Button
        outlined={true}
        large={true}
        onClick={() => {
          setOpen(true);
        }}
      >
        Delete
      </Button>

      <Dialog
        icon="info-sign"
        onClose={() => {setOpen(true);}}
        className={this.props.data.themeName}
        title="Are you sure?"
        isOpen={open}
      >
        <div>
          <p>
            This action cannot be undone. This will permanently delete this
            bento bundle and may have unintended consequences.
          </p>
          <div>
            <Button onClick={() => {setOpen(true);}}>Close</Button>
          </div>
        </div>
      </Dialog>
    </div>
  );
};

export default BentoBundleDeleteConfirmation;
